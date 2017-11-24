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
	string path = "D:\\drawingDiary\\sample.jpg";
	vector<vector<Point> > contours; // Vector for storing contours
	vector<Vec4i> hierarchy;

	Mat original = imread(path);
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

	/*
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;
	for (size_t i = 0; i< contours.size(); i++) // iterate through each contour.
	{
		double area = contourArea(contours[i]);  //  Find the area of contour

		if (area > largest_area)
		{
			largest_area = area;
			largest_contour_index = i;               //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}
	*/

	//select bounding rect
	vector<Rect> boundingRectlist;
	vector<double> arealist;


	for (size_t i = 0; i < contours.size(); i++) {
		Rect nowRect = boundingRect(contours[i]);
		if (i == 0) {
			boundingRectlist.push_back(nowRect);
			arealist.push_back(nowRect.area());
		}
		else {
			int check = 0;

			for (int j = 0; j < boundingRectlist.size(); j++) {

				//check intersect
				bool intersects = ((nowRect & boundingRectlist[j]).area() > 0);

				if (intersects) {	//if intersect, compare size and put larger one
					double area = nowRect.area();
					if (area > arealist[j]) {
						arealist[j] = area;
						boundingRectlist[j] = nowRect;
					}
				}
				else if (boundingRectlist[j] != nowRect) {	//if not intersect, increase check
					check++;
				}
				if (check == boundingRectlist.size()) {	//when there are nothing intersection
					boundingRectlist.push_back(nowRect);	//add list
					arealist.push_back(nowRect.area());
				}
			}
		}
	}

	//delete duplication
	vector<Rect> deleteDup;
	for (int i = 0; i < boundingRectlist.size(); i++){
		if (i == 0) {
			deleteDup.push_back(boundingRectlist[i]);
		}
		else {
			for (int j = 0; j < deleteDup.size(); j++) {
				if (deleteDup[j] == boundingRectlist[i]) {
					break;
				}

				if (j == deleteDup.size()-1) {
					deleteDup.push_back(boundingRectlist[i]);
				}
			}
		}
	}
	std::cout << deleteDup.size() << endl;

	//show & save bounding rect
	for (int i = 0; i <deleteDup.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Mat cut(original, deleteDup[i]);
		imshow(to_string(i), cut);
		imwrite(to_string(i) + ".jpg", cut);
		//rectangle(original, boundingRectlist[i].tl(), boundingRectlist[i].br(), color, 2, 8, 0);

	}

	// Show in a window
	imshow("result", original);

	waitKey(0);
	return 0;
}