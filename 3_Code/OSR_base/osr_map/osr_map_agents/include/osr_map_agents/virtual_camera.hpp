#ifndef __EVL_VIRTUAL_CAMERA__
#define __EVL_VIRTUAL_CAMERA__

#include <opencv/cxcore.h>
#include "osr_map_agents/se3.hpp"

namespace evl
{
    /** @brief ī�޶� ���� �Ķ���� (Camera Param)
    */
    struct CameraParam
    {
        double fx, fy;
        double cx, cy;

        double k1, k2;          // radial distortion coefficient (Brown & Fisheye)
        double k3, k4;          // radial distortion coefficient (Fisheye)
        double p1, p2;          // tangential distortion coefficient (Brown)
        double w;               // radial distortion coefficient (FOV)

        CameraParam()
        {
            fx = fy = cx = cy = 0;
            k1 = k2 = k3 = k4 = 0;
            p1 = p2 = 0;
            w = 0;
        }
    };

    /** @brief ī�޶� �� ���� �������̽� (Camera Base)
    */
    class CameraBase
    {
    public:
        /** @brief �⺻ ������
        */
        CameraBase();

        /** @brief ���� �ְ���� coupled �ְ�𵨷� ���� (CameraBasic, CameraFisheye �⺻ ������)
        */
        void set_coupled_distortion_model();

        /** @brief ���� �ְ���� Decoupled �ְ�𵨷� ���� (CameraFOV �⺻ ������)
        */
        void set_decoupled_distortion_model();

        /** @brief FOV ���� EVL ī�޶� �Ķ����(����&�ܺ�)�� ���Ϸκ��� �о���δ�.
        @param file_path �Ķ���Ͱ� ����� ���ϸ� (evl_camera.yml)
        @param cam_id ��� ī�޶� ID ���ڿ� (CAM111, CAM121, ...)
        @return ������ true ��ȯ
        */
        bool load_evlcamera(const char* file_path, const char* cam_id);

        /** @brief ī�޶� ���� �Ķ���͸� ����
        @param param ī�޶� ���� �Ķ���� ����ü
        */
        void set_intrinsic_parameters(const CameraParam& param);

        /** @brief ���� ������ ī�޶� ���� �Ķ���͸� ��ȯ
        @return ī�޶� ���� �Ķ���� ����ü
        */
        CameraParam get_intrinsic_parameters() const;

        /** @brief ī�޶� �ܺ� �Ķ���͸� ���� (3D �ڼ�����)
        @param param ī�޶� �ܺ� �Ķ���� ����ü
        */
        void set_extrinsic(const SE3& se3);

        /** @brief ���� ������ ī�޶� �ܺ� �Ķ���͸� ��ȯ (3D �ڼ�����)
        @return ī�޶� �ܺ� �Ķ���� ����ü
        */
        SE3 get_extrinsic() const;

        /** @brief ������ǥ�� ī�޶� �̹����� ������ �ȼ���ǥ�� ��ȯ
        @param src 3D ������ǥ (�Է�)
        @param dst 2D �̹��� �ȼ���ǥ (���)
        @return �̹��� ������ �Ұ����� ���(depth<=0) false ��ȯ
        */
        bool project(const cv::Point3d& src, cv::Point2d& dst) const;

        /** @brief ������ǥ�� ī�޶� �̹����� ������ �ȼ���ǥ�� ��ȯ
        @param src 3D ������ǥ (�Է�)
        @param dst 2D �̹��� �ȼ���ǥ (���)
        @param depth �Է� ������ǥ�� depth
        @return �̹��� ������ �Ұ����� ���(depth<=0) false ��ȯ
        */
        bool project(const cv::Point3d& src, cv::Point2d& dst, double& depth) const;

        /** @brief ������ǥ�� ī�޶� �̹����� ������ �ȼ���ǥ�� ��ȯ
        @param src 3D ������ǥ (�Է�)
        @param dst 2D �̹��� �ȼ���ǥ (���)
        */
        void project(const std::vector<cv::Point3d>& src, std::vector<cv::Point2d>& dst) const;

        /** @brief ������ǥ�� ī�޶� �̹����� ������ �ȼ���ǥ�� ��ȯ
        @param src 3D ������ǥ (�Է�)
        @param dst 2D �̹��� �ȼ���ǥ (���)
        @param depth �Է� ������ǥ�� depth
        */
        void project(const std::vector<cv::Point3d>& src, std::vector<cv::Point2d>& dst, std::vector<double>& depth) const;

        /** @brief �̹��� �ȼ���ǥ�� �����Ǵ� ���� ������ǥ(X, Y)�� ��ȯ
        @param src 2D �̹��� �ȼ���ǥ (�Է�)
        @param dst 2D ���� ������ǥ X, Y (Z = 0�̹Ƿ� ����)
        @return ���� �������� �Ұ����� ���(ray�� �ϴ��� ���ϴ� ��� ��) false ��ȯ
        */
        bool unproject_ground(const cv::Point2d& src, cv::Point2d& dst) const; // src: ������ǥ, dst: ������ǥ(����)

        /** @brief �̹��� �ȼ���ǥ�� �����Ǵ� ���� ������ ������ ī�޶�κ����� ����� �Ÿ� �� ���� ��ȯ
        @param src 2D �̹��� �ȼ���ǥ (�Է�)
        @param distance ���� ���������� ī�޶�κ����� ����
        @param theta_radian ���� ���������� ����(ī�޶� ������ ���� ���� ���� +, ������ ���� -)
        @return ���� �������� �Ұ����� ���(ray�� �ϴ��� ���ϴ� ��� ��) false ��ȯ
        */
        bool unproject_ground_relative(const cv::Point2d& src, double& distance, double& theta_radian) const;

        /** @brief �̹��� �ȼ���ǥ�� �����Ǵ� ���� ������ǥ(X, Y)�� ��ȯ
        @param src 2D �̹��� �ȼ���ǥ (�Է�)
        @param dst 2D ���� ������ǥ X, Y (Z = 0�̹Ƿ� ����)
        @param valid ���� �������� �Ұ����� ���(ray�� �ϴ��� ���ϴ� ��� ��) false
        */
        void unproject_ground(const std::vector<cv::Point2d>& src, std::vector<cv::Point2d>& dst, std::vector<bool>& valid) const;

        /** @brief �̹��� �ȼ���ǥ�� �����Ǵ� ���� ������ ������ ī�޶�κ����� ����� �Ÿ� �� ���� ��ȯ
        @param src 2D �̹��� �ȼ���ǥ (�Է�)
        @param distance ���� ���������� ī�޶�κ����� ����
        @param theta_radian ���� ���������� ����(ī�޶� ������ ���� ���� ���� +, ������ ���� -)
        @param valid ���� �������� �Ұ����� ���(ray�� �ϴ��� ���ϴ� ��� ��) false
        */
        void unproject_ground_relative(const std::vector<cv::Point2d>& src, std::vector<double>& distance, std::vector<double>& theta_radian, std::vector<bool>& valid) const;

        /** @brief ����� �ְ��� �ݿ��� �̹����� ��ȯ
        @param src �Է� �̹���
        @param dst �ְ��� �ݿ��� �̹���
        */
        void distort_image(const cv::Mat& src, cv::Mat& dst) const;

        /** @brief ����� �ְ��� ������ �̹����� ��ȯ
        @param src �Է� �̹���
        @param dst �ְ� ������ �̹���
        @param dst_scale ���� �̹���(dst)�� ������ (���� Ŭ���� �ְ���� ���� �����߸��� �پ��. ���� ���� ������ ǥ��). 0 ������ ���� �ָ� �ְ���� ������ �߸��� �ּ�ȭ�ǵ��� �������� �ڵ����� ����
        @param dst_size ���� �̹���(dst)�� �̹��� �ػ�
        @param canonical_view Canonical View(tilt=0, roll=0�� ī�޶� ���� �̹���) ��ȯ ����
        */
        void undistort_image(const cv::Mat& src, cv::Mat& dst, double dst_scale = 1.0, double dst_size = 1.0, bool canonical_view = false);

        /** @brief ���� �̹��� ���� �𵨷� ��ȯ�� �̹����� ��ȯ
        @param src �Է� �̹���
        @param dst ���� �̹���
        */
        void spherical_image(const cv::Mat& src, cv::Mat& dst) const;

        /** @brief �Է� �ȼ���ǥ�� Canonical ��ǥ(tilt=roll=0)�� ��ȯ
        @param pt 2D �̹��� ��ǥ (�Է� & ���)
        */
        void pixel2canonical(cv::Point2d& pt) const;

        /** @brief �Է� �ȼ���ǥ�� Canonical ��ǥ(tilt=roll=0)�� ��ȯ
        @param pts 2D �̹��� ��ǥ (�Է� & ���)
        */
        void pixel2canonical(std::vector<cv::Point2d>& pts) const;

        /** @brief �Է� canonical ��ǥ(tilt=roll=0)�� �ȼ���ǥ�� ��ȯ
        @param pt 2D canonical ��ǥ (�Է� & ���)
        */
        void canonical2pixel(cv::Point2d& pt) const;

        /** @brief �Է� canonical ��ǥ(tilt=roll=0)�� �ȼ���ǥ�� ��ȯ
        @param pts 2D canonical ��ǥ (�Է� & ���)
        */
        void canonical2pixel(std::vector<cv::Point2d>& pts) const;

        /** @brief �Է� �ȼ���ǥ�� �����̹��� ��ǥ�� ��ȯ
        @param pt 2D �̹��� ��ǥ (�Է� & ���)
        */
        void normalize(cv::Point2d& pt) const;

        /** @brief �Է� �ȼ���ǥ�� �����̹��� ��ǥ�� ��ȯ
        @param pts 2D �̹��� ��ǥ (�Է� & ���)
        */
        void normalize(std::vector<cv::Point2d>& pts) const;

        /** @brief �Է� �ȼ���ǥ�� �����̹��� ��ǥ�� ��ȯ
        @param pt 2D �̹��� ��ǥ (�Է� & ���)
        */
        void denormalize(cv::Point2d& pt) const;

        /** @brief �Է� �ȼ���ǥ�� �����̹��� ��ǥ�� ��ȯ
        @param pts 2D �̹��� ��ǥ (�Է� & ���)
        */
        void denormalize(std::vector<cv::Point2d>& pts) const;

        /** @brief �Է� �ȼ���ǥ�� �ְ�� �ȼ���ǥ�� ��ȯ
        @param pt 2D �ȼ� ��ǥ (�Է� & ���)
        */
        void distort(cv::Point2d& pt) const;

        /** @brief �Է� �ȼ���ǥ�� �ְ�� �ȼ���ǥ�� ��ȯ
        @param pts 2D �ȼ� ��ǥ (�Է� & ���)
        */
        void distort(std::vector<cv::Point2d>& pts) const;

        /** @brief �Է� �ȼ���ǥ�� �ְ���� �ȼ���ǥ�� ��ȯ
        @param pt 2D �ȼ� ��ǥ (�Է� & ���)
        */
        void undistort(cv::Point2d& pt) const;

        /** @brief �Է� �ȼ���ǥ�� �ְ���� �ȼ���ǥ�� ��ȯ
        @param pts 2D �ȼ� ��ǥ (�Է� & ���)
        */
        void undistort(std::vector<cv::Point2d>& pts) const;

        /** @brief �Է� ���� �̹��� ��ǥ�� �ְ�� ���� �̹��� ��ǥ�� ��ȯ
        @param pts ���� �̹��� ��ǥ (�Է� & ���)
        */
        void distort_normal(cv::Point2d& pt) const;

        /** @brief �Է� ���� �̹��� ��ǥ�� �ְ�� ���� �̹��� ��ǥ�� ��ȯ
        @param pts ���� �̹��� ��ǥ (�Է� & ���)
        */
        void distort_normal(std::vector<cv::Point2d>& pts) const;

        /** @brief �Է� ���� �̹��� ��ǥ�� �ְ���� ���� �̹��� ��ǥ�� ��ȯ
        @param pts ���� �̹��� ��ǥ (�Է� & ���)
        */
        void undistort_normal(cv::Point2d& pt) const;

        /** @brief �Է� ���� �̹��� ��ǥ�� �ְ���� ���� �̹��� ��ǥ�� ��ȯ
        @param pts ���� �̹��� ��ǥ (�Է� & ���)
        */
        void undistort_normal(std::vector<cv::Point2d>& pts) const;

        bool cvtBox2Cylinder(cv::Point3d& p, cv::Size2d& sz, const cv::Rect& box, double ground_z_offset = 0) const;

        bool cvtBox2Cylinder(cv::Point2d& foot, cv::Point2d& head, cv::Point3d& p, cv::Size2d& sz, const cv::Rect& box, double ground_z_offset = 0) const;

    protected:
        CameraParam m_intrinsic;    // intrinsic parameters
        SE3 m_se3;                  // extrinsic parameters
        enum { DM_COUPLED, DM_DECOUPLED } m_distortion_model_type;  // distortion model type

        // lookup table for distortion correction & canonical view
        std::vector<cv::Point> m_lookup_pts;
        cv::Mat m_dst;
        CameraParam m_lookup_intrinsic;
        double m_dst_scale = 0;
        bool m_canonical_view = false;
        void _build_lookup_table(const cv::Mat& src, double dst_scale = 1.0, bool canonical_view = false);

        void _center_normalize(cv::Point2d& pt) const;
        void _center_normalize(std::vector<cv::Point2d>& pts) const;
        void _center_denormalize(cv::Point2d& pt) const;
        void _center_denormalize(std::vector<cv::Point2d>& pts) const;

        virtual void _distort(cv::Point2d& pt) const = 0;
        virtual void _distort(std::vector<cv::Point2d>& pts) const = 0;
        virtual void _undistort(cv::Point2d& pt) const = 0;
        virtual void _undistort(std::vector<cv::Point2d>& pts) const = 0;
    };


    /** @brief OpenCV �⺻ ī�޶� �� (Camera Basic)
    */
    class CameraBasic : public CameraBase
    {
    public:
        CameraBasic();
    protected:
        void _distort(cv::Point2d& pt) const;
        void _distort(std::vector<cv::Point2d>& pts) const;
        void _undistort(cv::Point2d& pt) const;
        void _undistort(std::vector<cv::Point2d>& pts) const;
    };


    /** @brief OpenCV Fisheye ī�޶� �� (Camera Fisheye)
    */
    class CameraFisheye : public CameraBase
    {
    public:
        CameraFisheye();
    protected:
        void _distort(cv::Point2d& pt) const;
        void _distort(std::vector<cv::Point2d>& pts) const;
        void _undistort(cv::Point2d& pt) const;
        void _undistort(std::vector<cv::Point2d>& pts) const;
    };


    /** @brief FOV ī�޶� �� (Camera FOV)
    */
    class CameraFOV : public CameraBase
    {
    public:
        CameraFOV();
    protected:
        void _distort(cv::Point2d& pt) const;
        void _distort(std::vector<cv::Point2d>& pts) const;
        void _undistort(cv::Point2d& pt) const;
        void _undistort(std::vector<cv::Point2d>& pts) const;
    };

} // End of 'evl'

#endif // End of '__EVL_VIRTUAL_CAMERA__'
