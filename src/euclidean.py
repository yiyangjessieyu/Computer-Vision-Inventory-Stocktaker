int main()
{
    cv::Mat input = cv::imread("../inputData/MultiLena.png");

    cv::Mat gray;
    cv::cvtColor(input, gray, CV_BGR2GRAY);

    cv::SiftFeatureDetector detector( 7500 );
    cv::SiftDescriptorExtractor describer;

    std::vector<cv::KeyPoint> keypoints;

    detector.detect( gray, keypoints );

    // draw keypoints
    cv::drawKeypoints(input,keypoints,input);



    cv::Mat descriptors;
    describer.compute(gray, keypoints, descriptors);

    int n = 4;
    std::vector<std::vector<float> > dists;
    std::vector<std::vector<int> > indices;

    // compute the N best matches between the descriptors and themselves.
    // REMIND: ONE best match will always be the keypoint itself in this setting!
    NBestMatches(descriptors, descriptors, n, dists, indices);

    for(unsigned int i=0; i<dists.size(); ++i)
    {
        for(unsigned int j=0; j<dists.at(i).size(); ++j)
        {
            if(dists.at(i).at(j) < 0.05)
                cv::line(input, keypoints[i].pt, keypoints[indices.at(i).at(j)].pt, cv::Scalar(255,255,255) );
        }
    }

    cv::imshow("input", input);
    cv::waitKey(0);

    return 0;
}
