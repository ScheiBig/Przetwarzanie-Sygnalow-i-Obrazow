import cv2
import time

def main():
	device = 0
	pos_frame = 0
	
	cap = cv2.VideoCapture(device)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	while not cap.isOpened():
		cap = cv2.VideoCapture(device)
		print("Waiting for video")
		k = cv2.waitKey(2000)
		if k == 27: return
	
	keep = True
	while keep:
		# t = time.time()
		flag, frame = cap.read()
		if flag:
			pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
			frm_gr = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
			frm_fl = cv2.GaussianBlur(frm_gr, (5, 5), 0)
			frm_ed = cv2.Canny(frm_fl, 60, 150)
			cv2.imshow("Frame", frame)
			cv2.imshow("Edge", frm_ed)
		else:
			cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
			print("Frame not ready")
			cv2.waitKey(100)
		
		if cv2.waitKey(1) == 27:
			keep = False
			cv2.destroyAllWindows()
			cap.release()
			break
		
		# t = time.time() - t
  
if __name__ == "__main__":
	main()
