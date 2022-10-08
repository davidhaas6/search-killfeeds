from vision import preprocess, get_contour_bboxs, get_img_text

from matplotlib import pyplot as plt
import cv2
import argparse
import os.path

DIR = "./test-data/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='OCR search sngine')
    subparsers = parser.add_subparsers()
    parser.add_argument(
        'image', 
        type=str, default=None, 
        required=False,
        help='Path for the input image of the OCR model'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', 
        help='Enable logging and visualizations'
    )

    dataset = subparsers.add_parser('dataset')
    dataset.add_argument(
        '-t', '--test',
        required=False,
        action='store_true', 
        help='Run the dataset module'
    )
    dataset.add_argument(
        '-m', '--make',
        required=False,
        action='store_true', 
        help='Create a dataset'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image:
        og_img = cv2.imread(os.path.join(DIR, args.image))

        if og_img is None: 
            return False

        # Use contours to identify rectangles holding the text
        img = preprocess(og_img)
        bboxes = get_contour_bboxs(img, True)
        
        # Read each killfeed line
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
