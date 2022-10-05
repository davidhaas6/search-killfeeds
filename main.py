from vision import preprocess, get_contour_bboxs, get_img_text

from matplotlib import pyplot as plt
import cv2
import pytesseract

DIR = "./test-data/"

def main():
    og_img = cv2.imread(DIR + "test_killfeed.png")
    img = preprocess(og_img)
    bboxes = get_contour_bboxs(img, True)

    i = 0
    for x, y, w, h in bboxes:
        crop = img[y : y + h, x : x + w]

        text = get_img_text(crop, min_confidence=30)
        print(text)
        
        plt.figure()
        plt.imshow(crop, cmap="binary")
        plt.title(text)
        cv2.imwrite(DIR + f"out_{i}.png", crop)
        i += 1


if __name__ == "__main__":
    main()
