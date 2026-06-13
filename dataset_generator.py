import os
import time
import librosa
import numpy as np
import pandas as pd
import torch

from demucs.pretrained import get_model
from demucs.apply import apply_model


SONGS_ROOT = "./songs"
OUTPUT_CSV = "results.csv"

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512

notes = [
    'C', 'C#', 'D', 'D#',
    'E', 'F', 'F#', 'G',
    'G#', 'A', 'A#', 'B'
]

major_profile = np.array([
    6.35, 2.23, 3.48, 2.33,
    4.38, 4.09, 2.52, 5.19,
    2.39, 3.66, 2.29, 2.88
])

minor_profile = np.array([
    6.33, 2.68, 3.52, 5.38,
    2.60, 3.53, 2.54, 4.75,
    3.98, 2.69, 3.34, 3.17
])


def detect_key(chroma_mean):

    major_scores = []
    minor_scores = []

    for i in range(12):

        rotated_major = np.roll(major_profile, i)
        rotated_minor = np.roll(minor_profile, i)

        major_corr = np.corrcoef(
            chroma_mean,
            rotated_major
        )[0, 1]

        minor_corr = np.corrcoef(
            chroma_mean,
            rotated_minor
        )[0, 1]

        major_scores.append(major_corr)
        minor_scores.append(minor_corr)

    best_major = np.argmax(major_scores)
    best_minor = np.argmax(minor_scores)

    major_score = major_scores[best_major]
    minor_score = minor_scores[best_minor]

    if major_score > minor_score:
        return (
            f"{notes[best_major]} Major",
            float(major_score)
        )

    return (
        f"{notes[best_minor]} Minor",
        float(minor_score)
    )


def detect_bpm(signal, sr):

    onset_env = librosa.onset.onset_strength(
        y=signal,
        sr=sr
    )

    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr
    )

    return float(np.atleast_1d(tempo)[0])


def get_chroma_mean(signal, sr):

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    return np.mean(chroma, axis=1)


def calculate_bpm_agreement(
        bpm_song,
        bpm_drums,
        bpm_hpss
):

    return float(
        np.std([
            bpm_song,
            bpm_drums,
            bpm_hpss
        ])
    )


def calculate_key_agreement(
        key_song,
        key_demucs,
        key_hpss
):

    keys = [
        key_song,
        key_demucs,
        key_hpss
    ]

    return max(
        keys.count(k)
        for k in set(keys)
    )



print("Loading Demucs model...")

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

model = get_model('htdemucs')
model.to(device)
model.eval()

print(f"Device: {device}")



dataset = []

for genre in sorted(os.listdir(SONGS_ROOT)):

    genre_path = os.path.join(
        SONGS_ROOT,
        genre
    )

    if not os.path.isdir(genre_path):
        continue

    print()
    print("=" * 60)
    print(f"GENRE: {genre}")
    print("=" * 60)

    for file in sorted(os.listdir(genre_path)):

        if not file.lower().endswith(
            (
                ".wav",
                ".mp3",
                ".flac"
            )
        ):
            continue

        song_path = os.path.join(
            genre_path,
            file
        )

        print(f"Processing: {file}")

        try:
            signal, sr = librosa.load(
                song_path,
                sr=SR,
                mono=True
            )

            signal = (
                signal /
                np.max(np.abs(signal))
            )

            duration = (
                len(signal) / sr
            )

            # DEMUCS
            wav, _ = librosa.load(
                song_path,
                sr=SR,
                mono=False
            )

            if wav.ndim == 1:
                wav = np.stack(
                    [wav, wav]
                )

            wav = torch.tensor(
                wav,
                dtype=torch.float32
            )

            wav = wav.unsqueeze(0)
            wav = wav.to(device)

            start_time = time.time()

            with torch.no_grad():
                sources = apply_model(
                    model,
                    wav
                )

            processing_time_demucs = (
                time.time() - start_time
            )

            sources = sources[0]

            drums_signal = (
                sources[0]
                .cpu()
                .numpy()
            )

            other_signal = (
                sources[2]
                .cpu()
                .numpy()
            )

            drums_signal = np.mean(
                drums_signal,
                axis=0
            )

            other_signal = np.mean(
                other_signal,
                axis=0
            )

            drums_signal /= (
                np.max(
                    np.abs(
                        drums_signal
                    )
                ) + 1e-8
            )

            other_signal /= (
                np.max(
                    np.abs(
                        other_signal
                    )
                ) + 1e-8
            )

            # HPSS
            harmonic_signal, percussive_signal = (
                librosa.effects.hpss(
                    signal
                )
            )

            # BPM
            bpm_song = detect_bpm(
                signal,
                sr
            )

            bpm_drums = detect_bpm(
                drums_signal,
                sr
            )

            bpm_hpss = detect_bpm(
                percussive_signal,
                sr
            )

            # SONG_KEY
            song_key, song_conf = (
                detect_key(
                    get_chroma_mean(
                        signal,
                        sr
                    )
                )
            )

            # KEY_DEMUCS
            demucs_key, demucs_conf = (
                detect_key(
                    get_chroma_mean(
                        other_signal,
                        sr
                    )
                )
            )

            # KEY_HPSS
            hpss_key, hpss_conf = (
                detect_key(
                    get_chroma_mean(
                        harmonic_signal,
                        sr
                    )
                )
            )


            agreement_bpm = (
                calculate_bpm_agreement(
                    bpm_song,
                    bpm_drums,
                    bpm_hpss
                )
            )

            agreement_key = (
                calculate_key_agreement(
                    song_key,
                    demucs_key,
                    hpss_key
                )
            )


            dataset.append({

                "song_name": file,
                "genre": genre,

                "duration": round(
                    duration,
                    2
                ),

                "sample_rate": sr,

                "processing_time_demucs":
                    round(
                        processing_time_demucs,
                        2
                    ),

                "bpm_song":
                    round(
                        bpm_song,
                        2
                    ),

                "bpm_drums":
                    round(
                        bpm_drums,
                        2
                    ),

                "bpm_hpss":
                    round(
                        bpm_hpss,
                        2
                    ),

                "key_song":
                    song_key,

                "confidence_song":
                    round(
                        song_conf,
                        4
                    ),

                "key_demucs":
                    demucs_key,

                "confidence_demucs":
                    round(
                        demucs_conf,
                        4
                    ),

                "key_hpss":
                    hpss_key,

                "confidence_hpss":
                    round(
                        hpss_conf,
                        4
                    ),

                "manual_bpm": "",

                "manual_key": "",

                "agreement_bpm":
                    round(
                        agreement_bpm,
                        2
                    ),

                "agreement_key":
                    agreement_key
            })

            print(f"[DONE] {file}")

        except Exception as e:

            print(
                f"ERROR: {file}"
            )

            print(e)


df = pd.DataFrame(dataset)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("=" * 60)
print(f"Saved: {OUTPUT_CSV}")
print(f"Songs analyzed: {len(df)}")
print("=" * 60)
