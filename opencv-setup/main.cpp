#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    Mat frame, bg_diff, obj_diff, bg_frame, obj_frame, canny_output;
    VideoCapture cap;
    
    // Variables for line detection
    float rho, theta, a, b, x0, y0, total_rho, total_theta;
    Point p1, p2;
    vector<Vec2f> lines;

    // Pixel count for background and object differences
    int bg_pixelCount, obj_pixelCount;
    bool moving_detected = false;
    double moving_detected_time = 0;
    bool line_detected = false;
    double line_detected_time = 0;

    // Region of Interest (ROI)
    Rect bg_rect(100, 100, 600, 100); 
    Rect obj_rect(260, 250, 200, 180); 
    Rect line_detect_rect(260, 250, 200, 180); 

    if (!cap.open("Project2_video.mp4")) {
        cout << "no such file!" << endl;
        waitKey(0);
        return -1;
    }

    int fps = cap.get(CAP_PROP_FPS);
    double last_update_time = 0;

    while (1) {
        cap >> frame;
        if (frame.empty()) {
            cout << "end of video" << endl;
            break;
        }

        // ROI frames for background and object
        Mat bg_roi_frame = frame(bg_rect);
        Mat obj_roi_frame = frame(obj_rect);

        double current_time = (double)getTickCount() / getTickFrequency();

        // Update background and object frames every 2 seconds
        if (current_time - last_update_time >= 2.0) {
            bg_frame = bg_roi_frame.clone();
            obj_frame = obj_roi_frame.clone();
            last_update_time = current_time;
        }

        // Ensure frame sizes match
        if (bg_frame.size() != bg_roi_frame.size() || obj_frame.size() != obj_roi_frame.size()) {
            cerr << "Error: Size mismatch between frames." << endl;
            break;
        }

        // Background subtraction and thresholding for background ROI
        absdiff(bg_frame, bg_roi_frame, bg_diff);
        cvtColor(bg_diff, bg_diff, COLOR_BGR2GRAY);
        threshold(bg_diff, bg_diff, 50, 255, THRESH_BINARY);

        // Background subtraction and thresholding for object ROI
        absdiff(obj_frame, obj_roi_frame, obj_diff);
        cvtColor(obj_diff, obj_diff, COLOR_BGR2GRAY);
        threshold(obj_diff, obj_diff, 50, 255, THRESH_BINARY);

        // Front car movement detection (Foreground object movement detection)
        bg_pixelCount = countNonZero(bg_diff); // Difference in background ROI
        obj_pixelCount = countNonZero(obj_diff); // Difference in object ROI

        // If object movement is detected, display "Start Moving!"
        if (bg_pixelCount < 4000 && obj_pixelCount > 9000 && !moving_detected) { 
            moving_detected = true;
            moving_detected_time = current_time; // Record the time movement was detected
        }

        // Display "Start Moving!" if the front car moves
        if (moving_detected) {
            if (current_time - moving_detected_time < 5) {
                putText(frame, "Start Moving!", Point(50, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            } else {
                moving_detected = false;
            }
        }

        // Lane change detection (Line detection for lane change)
        Mat line_roi_frame = frame(line_detect_rect);
        cvtColor(line_roi_frame, line_roi_frame, COLOR_BGR2GRAY);
        blur(line_roi_frame, line_roi_frame, Size(5, 5));
        GaussianBlur(line_roi_frame, line_roi_frame, Size(15, 15), 2.5);
        Canny(line_roi_frame, canny_output,  200, 225, 5);

        // Use HoughLines to detect lines in the ROI
        HoughLines(canny_output, lines, 1, CV_PI / 180, 70, 0, 0, CV_PI / 180 * -20, CV_PI / 180 * 20);

        // If lines are detected, draw them on the frame
        if (!lines.empty()) {
            total_rho = 0;
            total_theta = 0;
            for (int i = 0; i < lines.size(); i++) {
                rho = lines[i][0];
                theta = lines[i][1];

                total_rho += rho;
                total_theta += theta;
            }
            total_rho /= lines.size();
            total_theta /= lines.size();

            a = cos(total_theta);
            b = sin(total_theta);
            x0 = a * total_rho;
            y0 = b * total_rho;

            p1 = Point(cvRound(x0 + 1000 * (-b)) + line_detect_rect.x, cvRound(y0 + 1000 * a) + line_detect_rect.y);
            p2 = Point(cvRound(x0 - 1000 * (-b)) + line_detect_rect.x, cvRound(y0 - 1000 * a) + line_detect_rect.y);

            line_detected = true;
            line_detected_time = (double)getTickCount() / getTickFrequency();
        }

        // Display "Lane departure!" for 5 seconds if a lane change is detected
        if (line_detected) {
            double current_time = (double)getTickCount() / getTickFrequency();
            if (current_time - line_detected_time < 5) {  
                putText(frame, "Lane departure!", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            } else {
                line_detected = false;
            }
        }

        // Display the window
        imshow("Project2", frame);

        char key = (char)waitKey(1000 / fps);
        if (key == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
