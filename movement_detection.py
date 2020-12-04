import argparse
import datetime
import imutils
import time
import cv2
from imutils.convenience import grab_contours
from imutils.video import VideoStream


# ? Argunment parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video")
ap.add_argument("-a", "--min-area", type=int,
                default=500, help="minimum area size")
args = vars(ap.parse_args())

# ? If the video argument is NONE then we're reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    # time.sleep(2.0)
else:  # ? otherwise, we're reading from the provided file
    vs = cv2.VideoCapture(args["video"])

# ? This initializes the first frame on the video stream
firstFrame = None

# ? Loop over the frames of the video
while True:
    # ? Grab the current frame and initialize the occupied/unoccupied text
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"

    # ? If the frame could not be grabed then we reach the end of the video
    if frame is None:
        break

    # ? Resize the frame, convert it ro grayscale
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ? If the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # ? Compute the absolute difference between the current frame and the first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # ? Dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # ? Loop over the contours
    for c in cnts:
        # ? Ignore the contour if is too small
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # ? Compute the bounding box for the contour, draw it on the frame, and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (51, 51, 255), 5)
        text = "Occupied"

    # ? Draw the text and timestamps on the frame
    cv2.putText(frame, "Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 51, 255), 3)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (51, 51, 255), 1)

    # ? Show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # ? If the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

# ? Cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
