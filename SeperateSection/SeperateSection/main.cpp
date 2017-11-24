#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

RNG rng(12345);

int main() {
	string path = "D:\\drawingDiary\\diary_sample.jpg";
	vector<vector<Point> > contours; // Vector for storing contours
	vector<Vec4i> hierarchy;

	Mat im_in = imread(path, IMREAD_GRAYSCALE);

	Mat im_th;
	threshold(im_in, im_th, 230, 255, THRESH_BINARY_INV);

	// Floodfill from point (0, 0)
	Mat im_floodfill = im_th.clone();
	floodFill(im_floodfill, cv::Point(0, 0), Scalar(255));

	// Invert floodfilled image
	Mat im_floodfill_inv;
	bitwise_not(im_floodfill, im_floodfill_inv);

	// Combine the two images to get the foreground.
	Mat im_out = (im_th | im_floodfill_inv);


	//Find contours
	findContours(im_th, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // Find the contours in the image
	
	int largest_area = 0, secondary_area = 0;
	int largest_contour_index, secondary_contour_index = 0;
	Rect bounding_rect, bounding_rect2;
	for (size_t i = 0; i < contours.size(); i++) // iterate through each contour.
	{
		double area = boundingRect(contours[i]).area();  //  Find the area of contour

		if (area > secondary_area) {
			if (area >= largest_area) {
				largest_area = area;
				largest_contour_index = i;               //Store the index of largest contour
				bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
			}
			else {
				secondary_area = area;
				secondary_contour_index = i;
				bounding_rect2 = boundingRect(contours[i]);
			}	
		}
	}

	cout << "first" << bounding_rect.area()<< endl;
	cout << "second : "<< bounding_rect2.area() << endl;
	
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	rectangle(im_in, bounding_rect2.tl(), bounding_rect2.br(), color, 2, 8, 0);
	imshow("result", im_in);

	waitKey(0);

	return 0;
}