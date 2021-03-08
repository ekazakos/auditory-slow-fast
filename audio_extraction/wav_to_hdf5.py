import os
import h5py
import argparse
import multiprocessing as mp
import librosa


def load_audio(root, file, target_sampling_rate):
    samples, sampling_rate = librosa.core.load(os.path.join(root, file),
                                               sr=None,
                                               mono=False)
    assert sampling_rate == target_sampling_rate, \
        "Sampling rate of audio files should be {} ({})".format(target_sampling_rate, file)
    assert len(samples.shape) == 1, "Audio files should be mono ({})".format(file)
    return samples, file.split('.')[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('audio_dir', help='Directory of EPIC-KITCHENS audio')
    parser.add_argument('output_dir', help='Directory to save the HDF5 audio dataset')
    parser.add_argument('--sampling_rate', type=int, default=24000, help='Sampling rate of audio in EPIC-KITCHENS')
    parser.add_argument('--processes', type=int, default=40, help='Number of processes for multiprocessing')

    args = parser.parse_args()

    f = h5py.File(args.output_dir, 'w')

    process_list = []
    pool = mp.Pool(processes=args.processes)
    for fn in os.listdir(args.audio_dir):
        if fn.endswith('.wav'):
            p = pool.apply_async(load_audio, (args.audio_dir, fn, args.sampling_rate))
            process_list.append(p)

    for p in process_list:
        samples, video_name = p.get()
        print(video_name)
        dset = f.create_dataset(video_name, data=samples)
    pool.terminate()
    pool.join()
    f.close()
