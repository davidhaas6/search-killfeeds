from matplotlib import pyplot as plt
import cv2
import numpy as np
import pytesseract
import os
import tqdm
from imutils.object_detection import non_max_suppression


def preprocess(img, invert=False):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if invert:
        img = cv2.bitwise_not(img)
    return img


def batch_preprocess(paths, out_dir=None, invert=False):
    imgs = [preprocess(cv2.imread(p), invert) for p in paths]

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for i, im in tqdm.tqdm(enumerate(imgs)):
            file_path = os.path.join(out_dir, f"{i}.png")
            cv2.imwrite(file_path, im)

    return imgs


def crop_topright_quadrant(img):
    (
        h,
        w,
    ) = img.shape[:2]
    return img[0 : h // 2, w // 2 : w]


def get_contour_bboxs(gray: np.ndarray, verbose=False):
    contours = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    bboxes = []
    if verbose:
        output = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        if peri > 500:
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                bboxes.append(cv2.boundingRect(approx))
                if verbose:
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (36, 255, 12), 2)

    if verbose:
        cv2.imwrite(f"out_contour.png", output)

    return bboxes


def detect_text(img):
    return cv2.dnn.TextDetectionModel.detect(img)


def get_img_text(img, min_confidence=0):
    df = pytesseract.image_to_data(img, output_type="data.frame")
    text = " ".join(df.text[df.conf > min_confidence])
    return text


class NNTextDetect:
    def __init__(self, model_path):
        self._net = cv2.dnn.readNet(model_path)

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        self._layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]


    def detect_text(self, rgb_img, min_confidence=0.5):
        rgb_img = self._resize_closest_const(rgb_img,32)
        scores, geom = self._predict(rgb_img)
        bboxes, conf = self._get_bboxes(geom, scores, min_confidence)
        return bboxes, conf


    def _predict(self, rgb_img):
        h, w = rgb_img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            rgb_img,
            1.0,
            (w, h),
            (123.68, 116.78, 103.94),
            swapRB=True,
        )
        self._net.setInput(blob)
        (scores, geometry) = self._net.forward(self._layer_names)
        return scores, geometry


    def _resize_closest_const(self, img, multiple=32):
        h, w = img.shape[:2]
        h1 = (h+multiple) - ((h+multiple) % multiple)
        w1 = (w+multiple) - ((w+multiple) % multiple)
        return cv2.resize(img,(w1,h1))


    def _get_bboxes(self, geometry, scores, min_confidence=0.5):
        (numRows, numCols) = geometry.shape[2:4]
        rects = []
        confidences = []

        for r in range(0, numRows):
            scoresData = scores[0, 0, r]
            boxData = geometry[0,:,r]
            # print(boxData)
            xData0 = geometry[0, 0, r]
            xData1 = geometry[0, 1, r]
            xData2 = geometry[0, 2, r]
            xData3 = geometry[0, 3, r]
            anglesData = geometry[0, 4, r]

            for c in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[c] < min_confidence:
                    continue

                (offsetX, offsetY) = (c * 4.0, r * 4.0)

                angle = anglesData[c]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[c] + xData2[c]
                w = xData1[c] + xData3[c]

                endX = int(offsetX + (cos * xData1[c]) + (sin * xData2[c]))
                endY = int(offsetY - (sin * xData1[c]) + (cos * xData2[c]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[c])
        
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        
        return boxes, confidences

    def draw_bboxes(self,img, bboxes, verbose=False):
        img = img.copy()
        for (startX, startY, endX, endY) in bboxes:
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        if verbose:
            cv2.imshow("Text Detection", img)
            cv2.waitKey(0)
        return img