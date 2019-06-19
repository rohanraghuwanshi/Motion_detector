import cv2,time

first_frame = None

video = cv2.VideoCapture(0)

while True:
    check,frame = video.read()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(25,25),0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    (_,cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for Contour in cnts:
        if cv2.contourArea(Contour)<10000:
            continue
        status=1
        (x, y, w, h)=cv2.boundingRect(Contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 5)

    cv2.imshow("Thresh Delta",thresh_frame)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(100)

    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows
