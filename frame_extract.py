import argparse
import cv2
#print(cv2.__version__)

def extract():
	success,image = vidcap.read()

	for i in range(1, 1000, 5):
	    vidcap.set(1,i-1)                      
	    success,image = vidcap.read(1)
	    frameId = vidcap.get(1)
	    cv2.imwrite("frame%d.jpg" % frameId, image)
	    print("Completed frame", i)

	vidcap.release()
	print("Complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #positional args
    parser.add_argument(
            '--open', nargs='?',
            type=str,
            required=True,
            help='path to video'
            )


    FLAGS = parser.parse_args()

    vidcap = cv2.VideoCapture('FLAGS.open')
    extract()