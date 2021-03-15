#ifndef PPFMAP_POSE_HH__
#define PPFMAP_POSE_HH__

#include <Eigen/Geometry>
#include <pcl/correspondence.h>
#include <pcl/common/common_headers.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
namespace ppfmap {

/** \brief Represents a pose supported by a correspondence.
 */
struct Pose {
    int votes;
    Eigen::Affine3f t;
    pcl::Correspondence c;
};

struct PoseScores {
    int scores;
    Eigen::Affine3f t;
};

pcl::PointCloud<pcl::PointNormal>::Ptr IcpTuning(pcl::PointCloud<pcl::PointNormal>::Ptr source, pcl::PointCloud<pcl::PointNormal>::Ptr target) 
{
    
    pcl::IterativeClosestPoint<pcl::PointNormal,pcl::PointNormal> icp;
    
    icp.setInputSource(source);
    icp.setInputTarget(target);
    //icp.setMaxCorrespondenceDistance (0.05);
    icp.setMaximumIterations(30);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon (1e-8);

    pcl::PointCloud<pcl::PointNormal>::Ptr Final (new pcl::PointCloud<pcl::PointNormal>);

    icp.align(*Final);

    return Final;

}


bool similarPoses(
    const Eigen::Affine3f& pose1, 
    const Eigen::Affine3f& pose2,
    const float translation_threshold,
    const float rotation_threshold) {

    // Translation difference.
    float position_diff = (pose1.translation() - pose2.translation()).norm();
    
    // Rotation angle difference.
    Eigen::AngleAxisf rotation_diff_mat(pose1.rotation().inverse() * pose2.rotation());
    float rotation_diff = fabsf(rotation_diff_mat.angle());

    return position_diff < translation_threshold &&
           rotation_diff < rotation_threshold;
}

void clusterPoses(
    const std::vector<Pose>& poses, 
    const float translation_threshold,
    const float rotation_threshold,
    Eigen::Affine3f &trans, 
    pcl::Correspondences& corr, 
    int& votes) {

    int cluster_idx;
    std::vector<std::pair<int, int> > cluster_votes;
    std::vector<std::vector<Pose> > pose_clusters;

    for (const auto& pose : poses) {

        bool found_cluster = false;

        cluster_idx = 0;
        for (auto& cluster : pose_clusters) {
            if (similarPoses(pose.t, cluster.front().t, translation_threshold, rotation_threshold)) {
                found_cluster = true;
                cluster.push_back(pose);
                cluster_votes[cluster_idx].first += pose.votes;
            }
            ++cluster_idx;
        }

        // Add a new cluster of poses
        if (found_cluster == false) {
            std::vector<Pose> new_cluster;
            new_cluster.push_back(pose);
            pose_clusters.push_back(new_cluster);
            cluster_votes.push_back(std::pair<int, int>(pose.votes , pose_clusters.size() - 1));
        }
    }

    std::sort(cluster_votes.begin(), cluster_votes.end());

    Eigen::Vector3f translation_average (0.0, 0.0, 0.0);
    Eigen::Vector4f rotation_average (0.0, 0.0, 0.0, 0.0);

    for (const auto& pose : pose_clusters[cluster_votes.back().second]) {
        translation_average += pose.t.translation();
        rotation_average += Eigen::Quaternionf(pose.t.rotation()).coeffs();
        corr.push_back(pose.c);
    }

    translation_average /= static_cast<float> (pose_clusters[cluster_votes.back().second].size());
    rotation_average /= static_cast<float> (pose_clusters[cluster_votes.back().second].size());

    trans.translation().matrix() = translation_average;
    trans.linear().matrix() = Eigen::Quaternionf(rotation_average).normalized().toRotationMatrix();

    votes = cluster_votes.back().first;
}

template <typename PointT,typename NormalT>
void clusterPoses(
    const std::vector<Pose>& poses, 
    typename pcl::PointCloud<PointT>::Ptr modelCloud,
    typename pcl::PointCloud<PointT>::Ptr sceneCloud,
    typename pcl::PointCloud<NormalT>::Ptr modelNormals,
    const float translation_threshold,
    const float rotation_threshold,
    const float clustering_threshold,
    const float neighborhood_percentage,
    const float model_diameter,
    Eigen::Vector3f& cameraPosition,
    Eigen::Affine3f &trans, 
    pcl::Correspondences& corr, 
    int& votes) {

    int cluster_idx;
    std::vector<std::pair<int, int> > cluster_votes;
    std::vector<std::vector<Pose> > pose_clusters;

    for (const auto& pose : poses) {

        bool found_cluster = false;

        cluster_idx = 0;
        for (auto& cluster : pose_clusters) {
            if (similarPoses(pose.t, cluster.front().t, translation_threshold, rotation_threshold)) {
                found_cluster = true;
                cluster.push_back(pose);
                cluster_votes[cluster_idx].first += pose.votes;
            }
            ++cluster_idx;
        }

        // Add a new cluster of poses
        if (found_cluster == false) {
            std::vector<Pose> new_cluster;
            new_cluster.push_back(pose);
            pose_clusters.push_back(new_cluster);
            cluster_votes.push_back(std::pair<int, int>(pose.votes , pose_clusters.size() - 1));
        }
    }

    std::sort(cluster_votes.begin(), cluster_votes.end());

    std::vector<Eigen::Affine3f > poses_threshold;
    for (int i = cluster_votes.size() - 1; i >= 0; --i )
    {
        Eigen::Vector3f translation_average (0.0, 0.0, 0.0);
        Eigen::Vector4f rotation_average (0.0, 0.0, 0.0, 0.0);
        Eigen::Affine3f T_thresh;
        for (const auto& pose : pose_clusters[cluster_votes[i].second]) {
            if (i < (cluster_votes.size()-10))
                break;
            translation_average += pose.t.translation();
            rotation_average += Eigen::Quaternionf(pose.t.rotation()).coeffs();
            corr.push_back(pose.c);
        }
        translation_average /= static_cast<float> (pose_clusters[cluster_votes.back().second].size());
        rotation_average /= static_cast<float> (pose_clusters[cluster_votes.back().second].size());

        T_thresh.translation().matrix() = translation_average;
        T_thresh.linear().matrix() = Eigen::Quaternionf(rotation_average).normalized().toRotationMatrix();
        poses_threshold.push_back(T_thresh);

    }
    std::vector<std::pair<int, int> > cluster_scores;
    //std::cout<<"poses size:"<<poses_threshold.size()<<std::endl;
    int index = 0;
    for (const auto& pose : poses_threshold) {
        trans.translation().matrix()= pose.translation();
        trans.linear().matrix()= Eigen::Quaternionf(Eigen::Quaternionf(pose.rotation()).coeffs()).normalized().toRotationMatrix();
        typename pcl::PointCloud<PointT>::Ptr modelTransformed (new typename pcl::PointCloud<PointT>());
        typename pcl::PointCloud<NormalT>::Ptr modelTransformedN (new typename pcl::PointCloud<PointT>());
        //pcl::copyPointCloud(*modelCloud,*modelTransformed);
        pcl::transformPointCloud(*modelCloud, *modelTransformed, trans);
        pcl::transformPointCloud(*modelNormals, *modelTransformedN, trans);
        //modelTransformed_icp = IcpTuning(modelTransformed,sceneCloud,trans);
        pcl::KdTreeFLANN<PointT> scene_search;
        scene_search.setInputCloud(sceneCloud);
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        int score = 0;
        typename pcl::PointCloud<PointT>::iterator itP = modelTransformed->begin();
        typename pcl::PointCloud<NormalT>::iterator itN = modelTransformedN->begin();
        
        while (itP != modelTransformed->end() && itN != modelTransformedN->end())
        {   
            const auto& point = *itP;
            const auto& normal = *itN;
            
            if (( normal.getVector3fMap().transpose()*(cameraPosition -point.getVector3fMap()))<=0)
            {
                modelTransformed->erase(itP);
            }
            if (itP != modelTransformed->end())
                ++itP;
            if (itP != modelTransformedN->end())
                ++itN;
        }
        
        //modelTransformed_icp = IcpTuning(modelTransformed,sceneCloud,trans);
        for (std::size_t j = 0; j < modelTransformed->points.size (); ++j)
        {   
           
            if (scene_search.radiusSearch(modelTransformed->points[j], neighborhood_percentage * model_diameter, pointIdxRadiusSearch, pointRadiusSquaredDistance))
                {
                    
                    float minDist = *std::min_element(pointRadiusSquaredDistance.begin(),pointRadiusSquaredDistance.end());
                    float maxDist = *std::max_element(pointRadiusSquaredDistance.begin(),pointRadiusSquaredDistance.end());
                    //std::cout << "min and max distance are :"<<minDist<<"   "<<maxDist<<"  "<<index <<std::endl;
                    //std::vector<float>::iterator min_idx = std::min_element(pointNKNSquaredDistance.begin(),pointNKNSquaredDistance.end());
                    //int min_element_index = std::distance(pointNKNSquaredDistance.begin(),min_idx);
                    if (minDist<clustering_threshold*model_diameter  )
                    {   score++; 
                        
                    }

                }
        }
        
        
        cluster_scores.push_back(std::pair<int, int>(score , index));
        index++;

    }
    std::sort(cluster_scores.begin(), cluster_scores.end());
    // for (const auto& i :cluster_scores )
    //     std::cout<<"cluster_scores are :"<<i.first<<" , "<<i.second<<std::endl;
    Eigen::Affine3f resultPose = poses_threshold[cluster_scores.back().second];
    Eigen::Vector3f translation_result ;
    Eigen::Vector4f rotation_result ;
    translation_result = resultPose.translation();
    rotation_result = Eigen::Quaternionf(resultPose.rotation()).coeffs();

    // translation_average /= static_cast<float> (pose_clusters[cluster_votes.back().second].size());
    // rotation_average /= static_cast<float> (pose_clusters[cluster_votes.back().second].size());

    trans.translation().matrix() = translation_result;
    trans.linear().matrix() = Eigen::Quaternionf(rotation_result).normalized().toRotationMatrix();

    votes = cluster_votes.back().first;
}

} // namespace ppfmap

#endif // PPFMAP_POSE_HH__
