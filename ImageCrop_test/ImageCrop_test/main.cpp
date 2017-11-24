#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	string path = "D:\\drawingDiary\\dog.jpg";
	
	// Read image
	Mat im_in = imread(path, IMREAD_GRAYSCALE);
	Mat original = imread(path);

	// Threshold
	// Set values equal to or above 250 to 0.
	//thresholde 깔끔한 이미지는 220, 색이 연하거나 듬성듬성 칠해진 이미지는 240정도
	Mat im_th;
	threshold(im_in, im_th, 220, 255, THRESH_BINARY_INV);

	// Floodfill from point (0, 0)
	Mat im_floodfill = im_th.clone();
	floodFill(im_floodfill, cv::Point(0, 0), Scalar(255));

	// Invert floodfilled image
	Mat im_floodfill_inv;
	bitwise_not(im_floodfill, im_floodfill_inv);

	// Combine the two images to get the foreground.
	Mat im_out = (im_th | im_floodfill_inv);

	//Check mask and change alpha
	Mat result;
	cvtColor(original, result, CV_BGR2BGRA);

	for (int y = 0; y < im_out.rows; ++y) {
		for (int x = 0; x < im_out.cols; ++x) {
			uchar pixel_mask = im_out.at<uchar>(y, x);
			Vec4b & pixel_result = result.at<Vec4b>(y, x);
			if (pixel_mask == 0) {		//if mask pixel's color is white
				pixel_result[3] = 0;	//change original pixel's alpha 0
			}
		}
	}


	// Display images
	imshow("threshold", im_th);
	imshow("floodfill", im_floodfill);
	imshow("inverse", im_floodfill_inv);
	imshow("mask", im_out);
	imshow("result", result);

	// Save image
	imwrite("./reult_dog.png", result);

	waitKey(0);
	return 0;
}

/*
//흰 색상 투명하게
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
	//read image
	string filepath = "D:\\drawingDiary\\bird.jpg";
	Mat src = imread(filepath, 1);

	Mat input_bgra;
	cvtColor(src, input_bgra, CV_BGR2BGRA);

	for (int y = 0; y < input_bgra.rows; ++y) {
		for (int x = 0; x < input_bgra.cols; ++x) {
			Vec4b & pixel = input_bgra.at<Vec4b>(y, x);

			if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255) {
				pixel[3] = 0;
			}
		}
	}

	imshow("result", input_bgra);
	imwrite("./bird.png", input_bgra);
	
	waitKey();
	return 0;
}*/

/*
//얜 그냥 뭔가 잘 안됨..
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {

	// read in the apple (change path to the file)
	Mat img0 = imread("D:\\drawingDiary\\mice.jpg", 1);

	Mat img1;
	cvtColor(img0, img1, CV_RGB2GRAY);

	// apply your filter
	Canny(img1, img1, 100, 200);

	// find the contours
	vector< vector<Point> > contours;
	findContours(img1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// you could also reuse img1 here
	Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC1);

	// CV_FILLED fills the connected components found
	drawContours(mask, contours, -1, Scalar(255), CV_FILLED);

	

	//    vector<double> areas(contours.size());
	//    for(int i = 0; i < contours.size(); i++)
	//        areas[i] = contourArea(Mat(contours[i]));
	//    double max;
	//    Point maxPosition;
	//    minMaxLoc(Mat(areas),0,&max,0,&maxPosition);
	//    drawContours(mask, contours, maxPosition.y, Scalar(1), CV_FILLED);

	// let's create a new image now
	Mat crop(img0.rows, img0.cols, CV_8UC3);

	// set background to green
	crop.setTo(Scalar(0, 255, 0));

	// and copy the magic apple
	img0.copyTo(crop, mask);

	// normalize so imwrite(...)/imshow(...) shows the mask correctly!
	normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

	// show the images
	imshow("original", img0);
	imshow("mask", mask);
	imshow("canny", img1);
	imshow("cropped", crop);

	imwrite("./canny.jpg", img1);
	imwrite("./mask.jpg", mask);
	imwrite("./cropped.jpg", crop);

	waitKey();
	return 0;
}
*/
/*
//object를 지워버리는데 이걸 뒤집을 방법을 못찾겠음
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

namespace {
	Scalar bgr2ycrcb(Scalar bgr)
	{
		double R = bgr[2];
		double G = bgr[1];
		double B = bgr[0];
		double delta = 128; // Note: change this value if image type isn't CV_8U.

		double Y = 0.299 * R + 0.587 * G + 0.114 * B;
		double Cr = (R - Y) * 0.713 + delta;
		double Cb = (B - Y) * 0.564 + delta;

		return Scalar(Y, Cr, Cb, 0);
	}

	Mat1b chromaKey(const Mat3b & imageBGR, Scalar chromaBGR, double tInner, double tOuter)
	{
		assert(tInner <= tOuter);

		// Convert to YCrCb.
		assert(!imageBGR.empty());
		Size imageSize = imageBGR.size();
		Mat3b imageYCrCb;
		cvtColor(imageBGR, imageYCrCb, COLOR_BGR2YCrCb);
		Scalar chromaYCrCb = bgr2ycrcb(chromaBGR); // Convert a single BGR value to YCrCb.

													   // Build the mask.
		Mat1b mask = Mat1b::zeros(imageSize);
		const Vec3d key(chromaYCrCb[0], chromaYCrCb[1], chromaYCrCb[2]);

		for (int y = 0; y < imageSize.height; ++y)
		{
			for (int x = 0; x < imageSize.width; ++x)
			{
				const Vec3d color(imageYCrCb(y, x)[0], imageYCrCb(y, x)[1], imageYCrCb(y, x)[2]);
				double distance = norm(key - color);

				if (distance < tInner)
				{
					// Current pixel is fully part of the background.
					mask(y, x) = 0;
				}
				else if (distance > tOuter)
				{
					// Current pixel is fully part of the foreground.
					mask(y, x) = 255;
				}
				else
				{
					// Current pixel is partially part both, fore- and background; interpolate linearly.
					// Compute the interpolation factor and clip its value to the range [0, 255].
					double d1 = distance - tInner;
					double d2 = tOuter - tInner;
					uint8_t alpha = static_cast< uint8_t >(255. * (d1 / d2));

					mask(y, x) = alpha;
				}
			}
		}

		return mask;
	}

	Mat3b replaceBackground(const Mat3b & image, const Mat1b & mask, Scalar bgColor)
	{
		Size imageSize = image.size();
		const Vec3b bgColorVec(bgColor[0], bgColor[1], bgColor[2]);
		Mat3b newImage(image.size());

		for (int y = 0; y < imageSize.height; ++y)
		{
			for (int x = 0; x < imageSize.width; ++x)
			{
				uint8_t maskValue = mask(y, x);

				if (maskValue >= 255)
				{
					newImage(y, x) = bgColorVec;
				}
				else if (maskValue <= 0)
				{
					newImage(y, x) = image(y, x);
				}
				else
				{
					double alpha = 1. / static_cast< double >(maskValue);
					newImage(y, x) = alpha * image(y, x) + (1. - alpha) * bgColorVec;
				}
			}
		}

		return newImage;
	}

}

int main()
{
	string inputFilename = "D:\\drawingDiary\\mice.jpg";
	string maskFilename = "./mask.png";
	string newBackgroundFilename = "./newBackground.png";

	// Load the input image.
	Mat3b input = imread(inputFilename, IMREAD_COLOR);

	if (input.empty())
	{
		std::cerr << "Input file <" << inputFilename << "> could not be loaded ... " << std::endl;

		return 1;
	}

	// Apply the chroma keying and save the output.
	Scalar chroma(0, 0, 0, 0);
	double tInner = 100.;
	double tOuter = 170.;
	Mat1b mask = chromaKey(input, chroma, tInner, tOuter);

	Mat3b newBackground = replaceBackground(input, mask, Scalar(0, 255, 0, 0));

	imwrite(maskFilename, mask);
	imwrite(newBackgroundFilename, newBackground);

	namedWindow("input");
	imshow("input", input);
	namedWindow("mask");
	imshow("mask", mask);
	namedWindow("new background");
	imshow("new background", newBackground);
	waitKey(0);

	return 0;
}
*/