#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/ml.hpp>
#include <time.h>
#include <fstream> 
#include <windows.h>
#include "opencv2/highgui/highgui.hpp"
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <direct.h>


void mostra(cv::Mat original, cv::Mat risultati){

	    cv::Point pentu1;
		cv::Point pentu2;
		pentu1.x = 180;
		pentu1.y = 125;
		pentu2.x = 360;
		pentu2.y = 105;
        cv::rectangle(risultati, pentu1, pentu2, cv::Scalar(239, 239, 239), -1);
		cv::Point pentu11;
		cv::Point pentu22;
		pentu11.x = 180;
		pentu11.y = 125;
		pentu22.x = 360;
		pentu22.y = 105;
        cv::rectangle(risultati, pentu11, pentu22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Entusiasmo", cv::Point(30, 120), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pentu111;
		cv::Point pentu222;
		pentu111.x = 180;
		pentu111.y = 125;
		pentu222.x = 182;
		pentu222.y = 105;
		cv::rectangle(risultati, pentu111, pentu222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 120), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pinter1;
		cv::Point pinter2;
		pinter1.x = 180;
		pinter1.y = 155;
		pinter2.x = 360;
		pinter2.y = 135;
        cv::rectangle(risultati, pinter1, pinter2, cv::Scalar(239, 239, 239), -1);
		cv::Point pinter11;
		cv::Point pinter22;
		pinter11.x = 180;
		pinter11.y = 155;
		pinter22.x = 360;
		pinter22.y = 135;
        cv::rectangle(risultati, pinter11, pinter22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Interesse", cv::Point(30, 150), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pinter111;
		cv::Point pinter222;
		pinter111.x = 180;
		pinter111.y = 155;
		pinter222.x = 182;
		pinter222.y = 135;
		cv::rectangle(risultati, pinter111, pinter222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 150), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);

		
		
		cv::Point psorp1;
		cv::Point psorp2;
		psorp1.x = 180;
		psorp1.y = 185;
		psorp2.x = 360;
		psorp2.y = 165;
        cv::rectangle(risultati, psorp1, psorp2, cv::Scalar(239, 239, 239), -1);
		cv::Point psorp11;
		cv::Point psorp22;
		psorp11.x = 180;
		psorp11.y = 185;
		psorp22.x = 360;
		psorp22.y = 165;
        cv::rectangle(risultati, psorp11, psorp22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Sorpresa", cv::Point(30, 180), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point psorp111;
		cv::Point psorp222;
		psorp111.x = 180;
		psorp111.y = 185;
		psorp222.x = 182;
		psorp222.y = 165;
		cv::rectangle(risultati, psorp111, psorp222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 180), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pcur1;
		cv::Point pcur2;
		pcur1.x = 180;
		pcur1.y = 215;
		pcur2.x = 360;
		pcur2.y = 195;
        cv::rectangle(risultati, pcur1, pcur2, cv::Scalar(239, 239, 239), -1);
		cv::Point pcur11;
		cv::Point pcur22;
		pcur11.x = 180;
		pcur11.y = 215;
		pcur22.x = 360;
		pcur22.y = 195;
        cv::rectangle(risultati, pcur11, pcur22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Curiosita'", cv::Point(30, 210), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pcur111;
		cv::Point pcur222;
		pcur111.x = 180;
		pcur111.y = 215;
		pcur222.x = 182;
		pcur222.y = 195;
		cv::rectangle(risultati, pcur111, pcur222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 210), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pconc1;
		cv::Point pconc2;
		pconc1.x = 180;
		pconc1.y = 245;
		pconc2.x = 360;
		pconc2.y = 225;
        cv::rectangle(risultati, pconc1, pconc2, cv::Scalar(239, 239, 239), -1);
		cv::Point pconc11;
		cv::Point pconc22;
		pconc11.x = 180;
		pconc11.y = 245;
		pconc22.x = 360;
		pconc22.y = 225;
        cv::rectangle(risultati, pconc11, pconc22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Concentrazione", cv::Point(30, 240), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pconc111;
		cv::Point pconc222;
		pconc111.x = 180;
		pconc111.y = 245;
		pconc222.x = 182;
		pconc222.y = 225;
		cv::rectangle(risultati, pconc111, pconc222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 240), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point patt1;
		cv::Point patt2;
		patt1.x = 180;
		patt1.y = 275;
		patt2.x = 360;
		patt2.y = 255;
        cv::rectangle(risultati, patt1, patt2, cv::Scalar(239, 239, 239), -1);
		cv::Point patt11;
		cv::Point patt22;
		patt11.x = 180;
		patt11.y = 275;
		patt22.x = 360;
		patt22.y = 255;
        cv::rectangle(risultati, patt11, patt22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Attenzione", cv::Point(30, 270), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point patt111;
		cv::Point patt222;
		patt111.x = 180;
		patt111.y = 275;
		patt222.x = 182;
		patt222.y = 255;
		cv::rectangle(risultati, patt111, patt222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 270), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pdel1;
		cv::Point pdel2;
		pdel1.x = 180;
		pdel1.y = 305;
		pdel2.x = 360;
		pdel2.y = 285;
        cv::rectangle(risultati, pdel1, pdel2, cv::Scalar(239, 239, 239), -1);
		cv::Point pdel11;
		cv::Point pdel22;
		pdel11.x = 180;
		pdel11.y = 305;
		pdel22.x = 360;
		pdel22.y = 285;
        cv::rectangle(risultati, pdel11, pdel22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Delusione", cv::Point(30, 300), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pdel111;
		cv::Point pdel222;
		pdel111.x = 180;
		pdel111.y = 305;
		pdel222.x = 182;
		pdel222.y = 285;
		cv::rectangle(risultati, pdel111, pdel222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 300), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pnoi1;
		cv::Point pnoi2;
		pnoi1.x = 180;
		pnoi1.y = 335;
		pnoi2.x = 360;
		pnoi2.y = 315;
        cv::rectangle(risultati, pnoi1, pnoi2, cv::Scalar(239, 239, 239), -1);
		cv::Point pnoi11;
		cv::Point pnoi22;
		pnoi11.x = 180;
		pnoi11.y = 335;
		pnoi22.x = 360;
		pnoi22.y = 315;
        cv::rectangle(risultati, pnoi11, pnoi22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Noia", cv::Point(30, 330), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pnoi111;
		cv::Point pnoi222;
		pnoi111.x = 180;
		pnoi111.y = 335;
		pnoi222.x = 182;
		pnoi222.y = 315;
		cv::rectangle(risultati, pnoi111, pnoi222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 330), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pperpl1;
		cv::Point pperpl2;
		pperpl1.x = 180;
		pperpl1.y = 365;
		pperpl2.x = 360;
		pperpl2.y = 345;
        cv::rectangle(risultati, pperpl1, pperpl2, cv::Scalar(239, 239, 239), -1);
		cv::Point pperpl11;
		cv::Point pperpl22;
		pperpl11.x = 180;
		pperpl11.y = 365;
		pperpl22.x = 360;
		pperpl22.y = 345;
        cv::rectangle(risultati, pperpl11, pperpl22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Perplessita'", cv::Point(30, 360), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pperpl111;
		cv::Point pperpl222;
		pperpl111.x = 180;
		pperpl111.y = 365;
		pperpl222.x = 182;
		pperpl222.y = 345;
		cv::rectangle(risultati, pperpl111, pperpl222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 360), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pfas1;
		cv::Point pfas2;
		pfas1.x = 180;
		pfas1.y = 395;
		pfas2.x = 360;
		pfas2.y = 375;
        cv::rectangle(risultati, pfas1, pfas2, cv::Scalar(239, 239, 239), -1);
		cv::Point pfas11;
		cv::Point pfas22;
		pfas11.x = 180;
		pfas11.y = 395;
		pfas22.x = 360;
		pfas22.y = 375;
        cv::rectangle(risultati, pfas11, pfas22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Fastidio", cv::Point(30, 390), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pfas111;
		cv::Point pfas222;
		pfas111.x = 180;
		pfas111.y = 395;
		pfas222.x = 182;
		pfas222.y = 375;
		cv::rectangle(risultati, pfas111, pfas222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 390), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pfru1;
		cv::Point pfru2;
		pfru1.x = 180;
		pfru1.y = 425;
		pfru2.x = 360;
		pfru2.y = 405;
        cv::rectangle(risultati, pfru1, pfru2, cv::Scalar(239, 239, 239), -1);
		cv::Point pfru11;
		cv::Point pfru22;
		pfru11.x = 180;
		pfru11.y = 425;
		pfru22.x = 360;
		pfru22.y = 405;
        cv::rectangle(risultati, pfru11, pfru22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Frustrazione", cv::Point(30, 420), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pfru111;
		cv::Point pfru222;
		pfru111.x = 180;
		pfru111.y = 425;
		pfru222.x = 182;
		pfru222.y = 405;
		cv::rectangle(risultati, pfru111, pfru222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, " 0.00%", cv::Point(240, 420), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);

		

	cv::Size sz1 = original.size();
    cv::Size sz2 = risultati.size();
    cv::Mat finale(sz1.height, sz1.width+sz2.width, CV_8UC3);
    original.copyTo(finale(cv::Rect(0, 0, sz1.width, sz1.height)));
    risultati.copyTo(finale(cv::Rect(sz1.width, 0, sz2.width, sz2.height)));
    imshow("Real Time", finale);

}


void mostra2(cv::Mat original,cv::Mat risultati,cv::String percentuale11,cv::String percentuale22,cv::String percentuale33,cv::String percentuale44,cv::String percentuale55,cv::String percentuale66,cv::String percentuale77,cv::String percentuale88,cv::String percentuale99,cv::String percentuale1010,cv::String percentuale00,float perc_entusiasmo,float perc_interesse,float perc_sorpresa,float perc_curiosita,float perc_concentrazione,float perc_attenzione,float perc_delusione,float perc_noia,float perc_perplessita,float perc_fastidio,float perc_frustrazione){

	cv::Point pentu1;
		cv::Point pentu2;
		pentu1.x = 180;
		pentu1.y = 125;
		pentu2.x = 360;
		pentu2.y = 105;
        cv::rectangle(risultati, pentu1, pentu2, cv::Scalar(239, 239, 239), -1);
		cv::Point pentu11;
		cv::Point pentu22;
		pentu11.x = 180;
		pentu11.y = 125;
		pentu22.x = 360;
		pentu22.y = 105;
        cv::rectangle(risultati, pentu11, pentu22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Entusiasmo", cv::Point(30, 120), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pentu111;
		cv::Point pentu222;
		pentu111.x = 180;
		pentu111.y = 125;
		pentu222.x = 180+perc_entusiasmo;
		pentu222.y = 105;
		cv::rectangle(risultati, pentu111, pentu222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale11, cv::Point(240, 120), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pinter1;
		cv::Point pinter2;
		pinter1.x = 180;
		pinter1.y = 155;
		pinter2.x = 360;
		pinter2.y = 135;
        cv::rectangle(risultati, pinter1, pinter2, cv::Scalar(239, 239, 239), -1);
		cv::Point pinter11;
		cv::Point pinter22;
		pinter11.x = 180;
		pinter11.y = 155;
		pinter22.x = 360;
		pinter22.y = 135;
        cv::rectangle(risultati, pinter11, pinter22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Interesse", cv::Point(30, 150), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pinter111;
		cv::Point pinter222;
		pinter111.x = 180;
		pinter111.y = 155;
		pinter222.x = 180+perc_interesse;
		pinter222.y = 135;
		cv::rectangle(risultati, pinter111, pinter222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale22, cv::Point(240, 150), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point psorp1;
		cv::Point psorp2;
		psorp1.x = 180;
		psorp1.y = 185;
		psorp2.x = 360;
		psorp2.y = 165;
        cv::rectangle(risultati, psorp1, psorp2, cv::Scalar(239, 239, 239), -1);
		cv::Point psorp11;
		cv::Point psorp22;
		psorp11.x = 180;
		psorp11.y = 185;
		psorp22.x = 360;
		psorp22.y = 165;
        cv::rectangle(risultati, psorp11, psorp22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Sorpresa", cv::Point(30, 180), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point psorp111;
		cv::Point psorp222;
		psorp111.x = 180;
		psorp111.y = 185;
		psorp222.x = 180+perc_sorpresa;
		psorp222.y = 165;
		cv::rectangle(risultati, psorp111, psorp222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale33, cv::Point(240, 180), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pcur1;
		cv::Point pcur2;
		pcur1.x = 180;
		pcur1.y = 215;
		pcur2.x = 360;
		pcur2.y = 195;
        cv::rectangle(risultati, pcur1, pcur2, cv::Scalar(239, 239, 239), -1);
		cv::Point pcur11;
		cv::Point pcur22;
		pcur11.x = 180;
		pcur11.y = 215;
		pcur22.x = 360;
		pcur22.y = 195;
        cv::rectangle(risultati, pcur11, pcur22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Curiosita'", cv::Point(30, 210), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pcur111;
		cv::Point pcur222;
		pcur111.x = 180;
		pcur111.y = 215;
		pcur222.x = 180+perc_curiosita;
		pcur222.y = 195;
		cv::rectangle(risultati, pcur111, pcur222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale44, cv::Point(240, 210), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pconc1;
		cv::Point pconc2;
		pconc1.x = 180;
		pconc1.y = 245;
		pconc2.x = 360;
		pconc2.y = 225;
        cv::rectangle(risultati, pconc1, pconc2, cv::Scalar(239, 239, 239), -1);
		cv::Point pconc11;
		cv::Point pconc22;
		pconc11.x = 180;
		pconc11.y = 245;
		pconc22.x = 360;
		pconc22.y = 225;
        cv::rectangle(risultati, pconc11, pconc22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Concentrazione", cv::Point(30, 240), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pconc111;
		cv::Point pconc222;
		pconc111.x = 180;
		pconc111.y = 245;
		pconc222.x = 180+perc_concentrazione;
		pconc222.y = 225;
		cv::rectangle(risultati, pconc111, pconc222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale55, cv::Point(240, 240), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point patt1;
		cv::Point patt2;
		patt1.x = 180;
		patt1.y = 275;
		patt2.x = 360;
		patt2.y = 255;
        cv::rectangle(risultati, patt1, patt2, cv::Scalar(239, 239, 239), -1);
		cv::Point patt11;
		cv::Point patt22;
		patt11.x = 180;
		patt11.y = 275;
		patt22.x = 360;
		patt22.y = 255;
        cv::rectangle(risultati, patt11, patt22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Attenzione", cv::Point(30, 270), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point patt111;
		cv::Point patt222;
		patt111.x = 180;
		patt111.y = 275;
		patt222.x = 180+perc_attenzione;
		patt222.y = 255;
		cv::rectangle(risultati, patt111, patt222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale66, cv::Point(240, 270), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pdel1;
		cv::Point pdel2;
		pdel1.x = 180;
		pdel1.y = 305;
		pdel2.x = 360;
		pdel2.y = 285;
        cv::rectangle(risultati, pdel1, pdel2, cv::Scalar(239, 239, 239), -1);
		cv::Point pdel11;
		cv::Point pdel22;
		pdel11.x = 180;
		pdel11.y = 305;
		pdel22.x = 360;
		pdel22.y = 285;
        cv::rectangle(risultati, pdel11, pdel22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Delusione", cv::Point(30, 300), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pdel111;
		cv::Point pdel222;
		pdel111.x = 180;
		pdel111.y = 305;
		pdel222.x = 180+perc_delusione;
		pdel222.y = 285;
		cv::rectangle(risultati, pdel111, pdel222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale77, cv::Point(240, 300), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pnoi1;
		cv::Point pnoi2;
		pnoi1.x = 180;
		pnoi1.y = 335;
		pnoi2.x = 360;
		pnoi2.y = 315;
        cv::rectangle(risultati, pnoi1, pnoi2, cv::Scalar(239, 239, 239), -1);
		cv::Point pnoi11;
		cv::Point pnoi22;
		pnoi11.x = 180;
		pnoi11.y = 335;
		pnoi22.x = 360;
		pnoi22.y = 315;
        cv::rectangle(risultati, pnoi11, pnoi22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Noia", cv::Point(30, 330), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pnoi111;
		cv::Point pnoi222;
		pnoi111.x = 180;
		pnoi111.y = 335;
		pnoi222.x = 180+perc_noia;
		pnoi222.y = 315;
		cv::rectangle(risultati, pnoi111, pnoi222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale88, cv::Point(240, 330), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pperpl1;
		cv::Point pperpl2;
		pperpl1.x = 180;
		pperpl1.y = 365;
		pperpl2.x = 360;
		pperpl2.y = 345;
        cv::rectangle(risultati, pperpl1, pperpl2, cv::Scalar(239, 239, 239), -1);
		cv::Point pperpl11;
		cv::Point pperpl22;
		pperpl11.x = 180;
		pperpl11.y = 365;
		pperpl22.x = 360;
		pperpl22.y = 345;
        cv::rectangle(risultati, pperpl11, pperpl22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Perplessita'", cv::Point(30, 360), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pperpl111;
		cv::Point pperpl222;
		pperpl111.x = 180;
		pperpl111.y = 365;
		pperpl222.x = 180+perc_perplessita;
		pperpl222.y = 345;
		cv::rectangle(risultati, pperpl111, pperpl222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale99, cv::Point(240, 360), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pfas1;
		cv::Point pfas2;
		pfas1.x = 180;
		pfas1.y = 395;
		pfas2.x = 360;
		pfas2.y = 375;
        cv::rectangle(risultati, pfas1, pfas2, cv::Scalar(239, 239, 239), -1);
		cv::Point pfas11;
		cv::Point pfas22;
		pfas11.x = 180;
		pfas11.y = 395;
		pfas22.x = 360;
		pfas22.y = 375;
        cv::rectangle(risultati, pfas11, pfas22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Fastidio", cv::Point(30, 390), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pfas111;
		cv::Point pfas222;
		pfas111.x = 180;
		pfas111.y = 395;
		pfas222.x = 180+perc_fastidio;
		pfas222.y = 375;
		cv::rectangle(risultati, pfas111, pfas222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale1010, cv::Point(240, 390), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



		cv::Point pfru1;
		cv::Point pfru2;
		pfru1.x = 180;
		pfru1.y = 425;
		pfru2.x = 360;
		pfru2.y = 405;
        cv::rectangle(risultati, pfru1, pfru2, cv::Scalar(239, 239, 239), -1);
		cv::Point pfru11;
		cv::Point pfru22;
		pfru11.x = 180;
		pfru11.y = 425;
		pfru22.x = 360;
		pfru22.y = 405;
        cv::rectangle(risultati, pfru11, pfru22, cv::Scalar(128, 128, 128), 0);
		putText(risultati, "Frustrazione", cv::Point(30, 420), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);
	    cv::Point pfru111;
		cv::Point pfru222;
		pfru111.x = 180;
		pfru111.y = 425;
		pfru222.x = 180+perc_frustrazione;
		pfru222.y = 405;
		cv::rectangle(risultati, pfru111, pfru222, cv::Scalar(0, 255, 0), -1);
		putText(risultati, percentuale00, cv::Point(240, 420), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,0), 1.5);



	cv::Size sz1 = original.size();
    cv::Size sz2 = risultati.size();
    cv::Mat finale(sz1.height, sz1.width+sz2.width, CV_8UC3);
    original.copyTo(finale(cv::Rect(0, 0, sz1.width, sz1.height)));
    risultati.copyTo(finale(cv::Rect(sz1.width, 0, sz2.width, sz2.height)));
    imshow("Real Time", finale);
       
}