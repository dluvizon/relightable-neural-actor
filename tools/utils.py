import os


def extract_multiview_video_ffmpeg(video_file, output_path, start_frame, num_frames,
                                   monochrome=False, qscale=2, ffmpeg_cmd="/usr/bin/ffmpeg"):
    bash_cmd = f"{ffmpeg_cmd} -nostdin"
    bash_cmd += f" -i {video_file} -qscale:v {qscale}"
    bash_cmd += f" -start_number {start_frame} -vframes {num_frames}"
    if monochrome:
        bash_cmd += f" -filter_complex \"extractplanes=y\""
    bash_cmd += f" {output_path}"

    print(f"Running: {bash_cmd}")
    os.system(bash_cmd)

    return


def write_2d_array_to_file(filename, arr):
    assert arr.ndim == 2, f'Invalid input arr shape {arr.shape} != (num_rows, num_cols)'
    num_rows, num_cols = arr.shape

    text = ''
    for r in range(num_rows):
        for c in range(num_cols):
            text += f'{arr[r, c]:.6f}'
            if c < num_cols - 1:
                text += ' '
        text += '\n'

    with open(filename, 'w') as fip:
        fip.write(text)