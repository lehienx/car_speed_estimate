import cv2
import dlib
import math
import time

car_detect = cv2.CascadeClassifier('car_detect_harrcascade.xml')
video = cv2.VideoCapture('highway.mp4')
video.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# バウンディングボックスのサイズ
f_width = 1280
f_height = 720

# pixel/m 1 pixel = 1m
pixels_per_meter = 1

# 物体検出用の引数定義
frame_idx = 0
car_number = 0

# 1秒単位フレーム数
fps = 0
# ライン
input_w = 1960
laser_line = 200
laser_line_color = (0, 0, 255)

carTracker = {}
carStartPosition = {}
carCurrentPosition = {}
speed = [None] * 1000


# トラッキング結果に車クラス以外削除
def remove_bad_tracker():
    global carTracker, carStartPosition, carCurrentPosition

    # 削除車両リスト
    delete_id_list = []

    # 削除車両リストセット
    for car_id in carTracker.keys():
        # conf tracking < 4
        if carTracker[car_id].update(image) < 4:
            delete_id_list.append(car_id)

    # 車両削除
    for car_id in delete_id_list:
        carTracker.pop(car_id, None)
        carStartPosition.pop(car_id, None)
        carCurrentPosition.pop(car_id, None)

    return


# 速度算出
def calculate_speed(startPosition, currentPosition, fps_in):
    global pixels_per_meter

    # 車間距離測定（ピクセル)
    distance_in_pixels = math.sqrt(
        math.pow(currentPosition[0] - startPosition[0], 2) + math.pow(currentPosition[1] - startPosition[1], 2))

    # 距離測定（m）
    distance_in_meters = distance_in_pixels / pixels_per_meter

    # 速度算出(m/s)
    speed_in_meter_per_second = distance_in_meters * fps_in
    # km/hに変換  1m/s = 3,6km/h → V(km/h) = V(m/s) * 3.6
    speed_in_kilometer_per_hour = speed_in_meter_per_second * 3.6

    return speed_in_kilometer_per_hour


while True:
    start_time = time.time()
    _, image = video.read()

    if image is None:
        break

    image = cv2.resize(image, (f_width, f_height))
    output_image = image.copy()

    frame_idx += 1
    remove_bad_tracker()

    # 10フレームごと物体検出
    if not (frame_idx % 10):

        # 車両を物体検知
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cars = car_detect.detectMultiScale(gray, 1.2, 13, 18, (24, 24))

        # 物体検出された車両
        for (_x, _y, _w, _h) in cars:
            x = int(_x)
            y = int(_y)
            w = int(_w)
            h = int(_h)

            # 車両の重点
            x_center = x + 0.5 * w
            y_center = y + 0.5 * h

            matchCarID = None
            # 車両検出数
            for carID in carTracker.keys():
                # 車両の位置
                trackedPosition = carTracker[carID].get_position()
                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())
                # バウンディングボックスの中心点を取る
                t_x_center = t_x + 0.5 * t_w
                t_y_center = t_y + 0.5 * t_h

                # 検知済みの車両かチェック
                if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (
                        x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
                    matchCarID = carID

            # 未検知の車両トラッキング
            if matchCarID is None:
                # トラッカー生成
                tracker = dlib.correlation_tracker()
                tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                carTracker[car_number] = tracker
                carStartPosition[car_number] = [x, y, w, h]

                car_number += 1

    # 車両座標位置更新
    for carID in carTracker.keys():
        trackedPosition = carTracker[carID].get_position()

        t_x = int(trackedPosition.left())
        t_y = int(trackedPosition.top())
        t_w = int(trackedPosition.width())
        t_h = int(trackedPosition.height())

        # 長方形を描画
        cv2.rectangle(output_image, (t_x, t_y), (t_x + t_w, t_y + t_h), (255, 0, 0), 4)
        carCurrentPosition[carID] = [t_x, t_y, t_w, t_h]

    # 1秒単位フレーム数再算出
    end_time = time.time()
    if not (end_time == start_time):
        fps = 1.0 / (end_time - start_time)

    # 速度算出
    for i in carStartPosition.keys():
        [x1, y1, w1, h1] = carStartPosition[i]
        [x2, y2, w2, h2] = carCurrentPosition[i]

        carStartPosition[i] = [x2, y2, w2, h2]

        # 車両が移動される場合、速度算出
        if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
            # 今の車両座標（Y）＜200の場合速度算出
            if (speed[i] is None or speed[i] == 0) and y2 < 200:
                speed[i] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)

            # トラッキングIDと速度表示
            if speed[i] is not None and y2 >= 200:
                cv2.putText(output_image, str(i + 1) + "car" + str(int(speed[i])) + " km/h",
                            (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                car_number += 1

    # ラインを引く
    cv2.line(output_image, (0, laser_line), (input_w, laser_line), laser_line_color, 2)
    cv2.putText(output_image, "Laser line", (10, laser_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, laser_line_color, 2)
    cv2.imshow('video', output_image)

    # Detect Q key
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
