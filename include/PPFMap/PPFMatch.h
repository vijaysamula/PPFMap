#ifndef PPFMAP_PPFMATCH_HH__
#define PPFMAP_PPFMATCH_HH__

#include <unordered_map>

#include <pcl/common/common_headers.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>

#include <PPFMap/utils.h>
#include <PPFMap/Pose.h>
#include <PPFMap/DiscretizedPPF.h>
#include <PPFMap/ppf_cuda_calls.h>
#include <iostream>


namespace ppfmap {

template <typename PointT, typename NormalT>
class PPFMatch {
public:
    typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
    typedef typename pcl::PointCloud<NormalT>::Ptr NormalsPtr;
    typedef boost::shared_ptr<PPFMatch<PointT, NormalT> > Ptr;

    /** \brief Constructor for the 
     *  \param[in] disc_dist Discretization distance for the point pairs.
     *  \param[in] disc_angle Discretization angle for the ppf features.
     */
    PPFMatch(const float disc_dist = 0.01f, 
             const float disc_angle = 12.0f / 180.0f * static_cast<float>(M_PI))
        : distance_step(disc_dist)
        , angle_step(disc_angle)
        , translation_threshold(0.7f)
        , rotation_threshold(30.0f / 180.0f * static_cast<float>(M_PI))
        , neighborhood_percentage(0.5f)
        , ref_point_indices(new std::vector<int>())
        , model_map_initialized(false)
        , use_indices(false) {}

    /** \brief Default destructor **/
    virtual ~PPFMatch() {}

    /** \brief Sets the percentage of the models diameter to use as maximum 
     * radius while searching pairs in the scene.
     *  \param[in] percent Float between 0 and 1 to represent the percentage of 
     *  the maximum radius possible when searching for the model in the secene.
     */
    void setMaxRadiusPercent(const float percent) {
        neighborhood_percentage = percent;
    }

    /** \brief Sets the discretization parameter for the PPF Map creation.
     *  \param[in] dist_disc Discretization distance step.
     *  \param[in] angle_disc Discretization angle step.
     */
    void setDiscretizationParameters(const float dist_disc,
                                     const float angle_disc) {
        distance_step = dist_disc;
        angle_step = angle_disc;
    }

    /** \brief Sets the translation and rotation thresholds for the pose 
     * clustering step.
     *  \param[in] translation_thresh Translation threshold.
     *  \param[in] rotation_thresh Rotation threshold.
     *   \param[in] clustering_thresh Rotation threshold.
     */
    void setPoseClusteringThresholds(const float translation_thresh,
                                     const float rotation_thresh,
                                     const float clustering_thresh) {
        translation_threshold = translation_thresh;
        rotation_threshold = rotation_thresh;
        clustering_threshold = clustering_thresh;
    }
    /** \brief Sets the translation and rotation thresholds for the pose 
     * clustering step.
     *  \param[in] translation_thresh Translation threshold.
     *  \param[in] rotation_thresh Rotation threshold.
     */
    void setPoseClusteringThresholds(const float translation_thresh,
                                     const float rotation_thresh
                                     ) {
        translation_threshold = translation_thresh;
        rotation_threshold = rotation_thresh;
        
    }

    /** \brief Construct the PPF search structures for the model cloud.
     *  
     *  The model cloud contains the information about the object that is going 
     *  to be detected in the scene cloud. The necessary information to build 
     *  the search structure are the points and normals from the object.
     *
     *  \param[in] model Object's point cloud.
     *  \param[in] normals Cloud with the normals of the object.
     */
    void setModelCloud(const PointCloudPtr& model, const NormalsPtr& normals);

    /** \brief Specify a vector of indices of points in the cloud to use as 
     * reference points for the detection task.
     *  \param[in] ind Shared pointer to a vector of indices.
     */
    void setReferencePointIndices(const pcl::IndicesPtr ind) {
        ref_point_indices = ind;
        use_indices = true;
    }

    /** \brief Search of the model in an scene cloud and returns the 
     * correspondences and the transformation to the scene.
     *
     *  \param[in] cloud Point cloud of the scene.
     *  \param[in] normals Normals of the scene cloud.
     *  \param[out] trans Affine transformation from to model to the scene.
     *  \param[out] correspondence Supporting correspondences from the scene to 
     *  the model.
     *  \return True if the object appears in the scene, false otherwise.
     */
    void detect(const PointCloudPtr cloud, const NormalsPtr normals, 
                Eigen::Affine3f& trans, 
                pcl::Correspondences& correspondences, 
                int& votes);
    void detect(const PointCloudPtr sceneCloud, const NormalsPtr sceneNormals,
                const PointCloudPtr modelCloud,
                const NormalsPtr modelNormals ,
                Eigen::Vector3f& cameraPosition, 
                Eigen::Affine3f& trans, 
                pcl::Correspondences& correspondences, 
                int& votes);

    /** \brief Search the given scene for the object and returns a vector with 
     * the poses sorted by the votes obtained in the Hough space.
     *  
     *  \param[in] cloud Pointer to the scene cloud where to look for the 
     *  object.
     *  \param[in] normals Pointer to the cloud containing the scene normals.
     */
    bool detect(const PointCloudPtr cloud, const NormalsPtr normals, 
                std::vector<Pose>& poses);

    void detect(const PointCloudPtr cloud, const NormalsPtr normals,
                pcl::Correspondences& correspondences, int votes_threshold);
private:

    struct VotePair {
        int model_i;
        float alpha_m;
        
        VotePair(const int m_i, const float a_m)
            : model_i(m_i), alpha_m(a_m) {}
    };

    bool model_map_initialized;
    bool use_indices;
    float distance_step;
    float angle_step;
    float translation_threshold;
    float rotation_threshold;
    float clustering_threshold;
    float neighborhood_percentage;
    float model_diameter;

    PointCloudPtr model_;
    NormalsPtr normals_;
    pcl::IndicesPtr ref_point_indices;

    std::unordered_multimap<DiscretizedPPF, VotePair> map;
}; // class PPFMatch

} // namespace ppfmap

#include <PPFMap/impl/PPFMatch.hpp>

#endif // PPFMAP_PPFMATCH_HH__
