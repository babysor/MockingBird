   return [m for m in metadata if m is not None]

 def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool):
     metadata = []
     for book_dir in speaker_dir.glob("*"):
         if no_alignments:
             # Gather the utterance audios and texts
             # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
             extensions = ["*.wav", "*.flac", "*.mp3"]
             for extension in extensions:
                 wav_fpaths = book_dir.glob(extension)

                 for wav_fpath in wav_fpaths:
                     # Load the audio waveform
                     wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
                     if hparams.rescale:
                         wav = wav / np.abs(wav).max() * hparams.rescaling_max

                     # Get the corresponding text
                     # Check for .txt (for compatibility with other datasets)
                     text_fpath = wav_fpath.with_suffix(".txt")
                     if not text_fpath.exists():
                         # Check for .normalized.txt (LibriTTS)
                         text_fpath = wav_fpath.with_suffix(".normalized.txt")
                         assert text_fpath.exists()
                     with text_fpath.open("r") as text_file:
                         text = "".join([line for line in text_file])
                         text = text.replace("\"", "")
                         text = text.strip()

                     # Process the utterance
                     metadata.append(_process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name),
                                                       skip_existing, hparams))
         else:
             # Process alignment file (LibriSpeech support)
             # Gather the utterance audios and texts
             try:
                 alignments_fpath = next(book_dir.glob("*.alignment.txt"))
                 with alignments_fpath.open("r") as alignments_file:
                     alignments = [line.rstrip().split(" ") for line in alignments_file]
             except StopIteration:
                 # A few alignment files will be missing
                 continue

             # Iterate over each entry in the alignments file
             for wav_fname, words, end_times in alignments:
                 wav_fpath = book_dir.joinpath(wav_fname + ".flac")
                 assert wav_fpath.exists()
                 words = words.replace("\"", "").split(",")
                 end_times = list(map(float, end_times.replace("\"", "").split(",")))

                 # Process each sub-utterance
                 wavs, texts = _split_on_silences(wav_fpath, words, end_times, hparams)
                 for i, (wav, text) in enumerate(zip(wavs, texts)):
                     sub_basename = "%s_%02d" % (wav_fname, i)
                     metadata.append(_process_utterance(wav, text, out_dir, sub_basename,
                                                       skip_existing, hparams))

     return [m for m in metadata if m is not None]

 # TODO: use original split func
 def _split_on_silences(wav_fpath, words, end_times, hparams):
     # Load the audio waveform
     wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
     if hparams.rescale:
         wav = wav / np.abs(wav).max() * hparams.rescaling_max

     words = np.array(words)
     start_times = np.array([0.0] + end_times[:-1])
     end_times = np.array(end_times)
     assert len(words) == len(end_times) == len(start_times)
     assert words[0] == "" and words[-1] == ""

     # Find pauses that are too long
     mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
     mask[0] = mask[-1] = True
     breaks = np.where(mask)[0]

     # Profile the noise from the silences and perform noise reduction on the waveform
     silence_times = [[start_times[i], end_times[i]] for i in breaks]
     silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
     noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
     if len(noisy_wav) > hparams.sample_rate * 0.02:
         profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
         wav = logmmse.denoise(wav, profile, eta=0)

     # Re-attach segments that are too short
     segments = list(zip(breaks[:-1], breaks[1:]))
     segment_durations = [start_times[end] - end_times[start] for start, end in segments]
     i = 0
     while i < len(segments) and len(segments) > 1:
         if segment_durations[i] < hparams.utterance_min_duration:
             # See if the segment can be re-attached with the right or the left segment
             left_duration = float("inf") if i == 0 else segment_durations[i - 1]
             right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
             joined_duration = segment_durations[i] + min(left_duration, right_duration)

             # Do not re-attach if it causes the joined utterance to be too long
             if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
                 i += 1
                 continue

             # Re-attach the segment with the neighbour of shortest duration
             j = i - 1 if left_duration <= right_duration else i
             segments[j] = (segments[j][0], segments[j + 1][1])
             segment_durations[j] = joined_duration
             del segments[j + 1], segment_durations[j + 1]
         else:
             i += 1

     # Split the utterance
     segment_times = [[end_times[start], start_times[end]] for start, end in segments]
     segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
     wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
     texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]

     # # DEBUG: play the audio segments (run with -n=1)
     # import sounddevice as sd
     # if len(wavs) > 1:
     #     print("This sentence was split in %d segments:" % len(wavs))
     # else:
     #     print("There are no silences long enough for this sentence to be split:")
     # for wav, text in zip(wavs, texts):
     #     # Pad the waveform with 1 second of silence because sounddevice tends to cut them early
     #     # when playing them. You shouldn't need to do that in your parsers.
     #     wav = np.concatenate((wav, [0] * 16000))
     #     print("\t%s" % text)
     #     sd.play(wav, 16000, blocking=True)
     # print("")

     return wavs, texts 
