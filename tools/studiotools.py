import numpy as np
from xml.etree import ElementTree as ET


def read_camera_calib_xml(calib_xml_filepath):
    tree = ET.parse(calib_xml_filepath)

    sensors = tree.find(".//sensors[@next_id]")
    cameras = tree.find(".//cameras[@next_id]")

    num_sensors = sensors.attrib['next_id']
    if num_sensors:
        num_sensors = int(num_sensors)

    num_cameras = cameras.attrib['next_id']
    if num_cameras:
        num_cameras = int(num_cameras)

    assert (num_sensors == num_cameras), (
        f"Invalid calibration '{calib_xml_filepath}' with "
        f"{num_sensors} sensors and {num_cameras} cameras!"
    )

    camera_dict = {}
    for i in range(num_cameras):
        sensor = sensors.find(f".//sensor[@id='{i}']")
        w = int(sensor.find(f"resolution").attrib['width'])
        h = int(sensor.find(f"resolution").attrib['height'])
        
        calib_w = float(sensor.find(f"calibration").find(f"resolution").attrib['width'])
        calib_h = float(sensor.find(f"calibration").find(f"resolution").attrib['height'])
        f = float(sensor.find(f"calibration").find(f"f").text)
        cx = float(sensor.find(f"calibration").find(f"cx").text) + calib_w / 2
        cy = float(sensor.find(f"calibration").find(f"cy").text) + calib_h / 2
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        distortion_bkp = {
            'b1': float(sensor.find(f"calibration").find(f"b1").text),
            'k1': float(sensor.find(f"calibration").find(f"k1").text),
            'k2': float(sensor.find(f"calibration").find(f"k2").text),
            'k3': float(sensor.find(f"calibration").find(f"k3").text),
            'p1': float(sensor.find(f"calibration").find(f"p1").text),
            'p2': float(sensor.find(f"calibration").find(f"p2").text),
        }

        camera = cameras.find(f".//camera[@id='{i}']")
        assert (int(camera.attrib['sensor_id']) == i), (
            f"Unexpected sensor_id {camera.attrib['sensor_id']} for camera {i}"
        )
        camera_label = camera.attrib['label']
        RT = np.array([
            float(x) for x in camera.find(f"transform").text.split()
        ], dtype=np.float32).reshape((4, 4)) # in mm
        
        camera_dict[i] = {
            'w': w,
            'h': h,
            'K': K,
            'RT': RT,
            'distortion_bkp': distortion_bkp,
            'camera_label': camera_label,
        }

    return camera_dict


def read_camera_calib_matrix_3xn(fip):
    x = []
    for _ in range(3):
        line = fip.readline()
        if line == '':
            break
        sline = line.split()
        x.append([ float(s) for s in sline[1:] ])

    return np.array(x, dtype=np.float32)


def read_camera_calib_frame_item(fip):
    frame = {}
    while True:
        line = fip.readline()
        if line == '':
            break
        sline = line.split()

        if sline[0] == 'distortion':
            assert (len(sline) == 6), (f'Error in "distortion" field, unexpected string "{line}"')
            frame['distortion'] = [ float(s) for s in sline[1:] ]

        if sline[0] in ['origin', 'up', 'right']:
            assert (len(sline) == 4), (f'Error in "{sline[0]}" field, unexpected string "{line}"')
            frame[sline[0]] = [ float(s) for s in sline[1:] ]

        if sline[0] == 'right':
            break # last element in a frame, jump out

    return frame


def read_camera_calib_camera_item(fip):
    camera = {}
    camera['frames'] = {}
    while True:
        line = fip.readline()
        if line == '':
            break
        sline = line.split()

        if sline[0] == 'frame':
            assert (len(sline) == 2), (f'Error in "frame" field, unexpected string "{line}"')
            camera['frames'][int(sline[1])] = read_camera_calib_frame_item(fip)

        if (sline[0] == '#') and (sline[1] in ['extrinsics', 'intrinsics']):
            camera[sline[1]] = read_camera_calib_matrix_3xn(fip)

        if (sline[0] == 'time'):
            break # last element in camera, jump out

    return camera


def read_camera_calib(calib_filepath):
    cameras = {}
    with open(calib_filepath, 'r') as fip:
        while True:
            line = fip.readline()
            if line == '':
                break
            sline = line.split()

            if sline[0] == 'camera':
                assert (len(sline) == 3), (f'Error in "camera" field, unexpected string "{line}"')
                cameras[int(sline[1])] = read_camera_calib_camera_item(fip)        

    return cameras


def read_settings_txt(settings_filepath):
    settings = {}
    with open(settings_filepath, 'r') as fip:
        while True:
            line = fip.readline()
            if line == '':
                break
            sline = line.split()
            if sline[0] in ['START_FRAMES', 'END_FRAMES', 'FRAME_RATE', 'TARGET_RES_U', 'TARGET_RES_V']:
                settings[sline[0]] = int(sline[1])

    return settings


def compute_camera_K_and_RT(cameras_txt, settings):
    W = settings['TARGET_RES_U']
    H = settings['TARGET_RES_V']
    K = {}
    RT = {}
    for cam_id, cam_item in cameras_txt.items():
        K[cam_id] = np.concatenate([W * cam_item['intrinsics'][:2, :3], [[0, 0, 1]]])
        RT[cam_id] = np.concatenate([cam_item['extrinsics'][:3, :4], [[0, 0, 0, 1]]])
        RT[cam_id] = np.linalg.inv(RT[cam_id])
        RT[cam_id][0:3, 3] /= 1000 # convert translation from mm to meters

    return K, RT


def get_start_end_frames(settings_txt_filepath):
    with open(settings_txt_filepath, 'r') as fip:
        lines = fip.readlines()

    start_frame = int(lines[2].split()[1])
    end_frame = int(lines[3].split()[1])

    return start_frame, end_frame
