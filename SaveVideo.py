import cv2
import os
import tempfile


DIR_Q_TABLE_CHARTS = os.path.join(tempfile.gettempdir(), 'qtable_charts')
print(f"Q Table Charts Directory: {DIR_Q_TABLE_CHARTS}")


def make_video():
    # windows:
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Linux
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))

    for i in range(10, 50001, 10):
        img_path = os.path.join(DIR_Q_TABLE_CHARTS, f"{i}.png")
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    

if __name__ == "__main__":
    make_video()

