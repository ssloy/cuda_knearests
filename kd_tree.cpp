#include "kd_tree.h"
#include <algorithm>


/**
 * \brief Computes the squared distance between two nd points.
 * \param[in] p1 a pointer to the coordinates of the first point
 * \param[in] p2 a pointer to the coordinates of the second point
 * \param[in] dim dimension (number of coordinates of the points)
 * \return the squared distance between \p p1 and \p p2
 * \tparam float the numeric type of the point coordinates
 */
inline float distance2(const float* p1, const float* p2, int dim) {
    float result = 0.0;
    for(int i = 0; i < dim; i++) {
        result += (p2[i] - p1[i])*(p2[i] - p1[i]);
    }
    return result;
}

/**
 * \brief Comparison functor used to
 * sort the point indices.
 */
class ComparePointCoord {
    public:
        /**
         * \brief Creates a new ComparePointCoord
         * \param[in] nb_points number of points
         * \param[in] points pointer to first point
         * \param[in] stride number of floats between two
         *  consecutive points in array (=dimension if point
         *  array is compact).
         * \param[in] splitting_coord the coordinate to compare
         */
        ComparePointCoord(
                int nb_points,
                const float* points,
                int stride,
                int splitting_coord
                ) :
            nb_points_(nb_points),
            points_(points),
            stride_(stride),
            splitting_coord_(splitting_coord) {
            }

        /**
         * \brief Compares to point indices (does the
         * indirection and coordinate lookup).
         * \param[in] i index of first point to compare
         * \param[in] j index of second point to compare
         * \return true if point \p i is before point \p j, false otherwise
         */
        bool operator() (int i, int j) const {
            assert(i < nb_points_);
            assert(j < nb_points_);
            return
                (points_ + i * stride_)[splitting_coord_] <
                (points_ + j * stride_)[splitting_coord_]
                ;
        }

    private:
        int nb_points_;
        const float* points_;
        int stride_;
        int splitting_coord_;
};

/****************************************************************************/


KdTree::KdTree(int dim) : bbox_min_(dim), bbox_max_(dim), dimension_(dim), nb_points_(0), stride_(0), points_(NULL) {
}

KdTree::~KdTree() {
}

void KdTree::set_points(int nb_points, const float* points, int stride) {
    nb_points_ = nb_points;
    points_ = points;
    stride_ = stride;

    int sz = max_node_index(1, 0, nb_points) + 1;

    point_index_.resize(nb_points);
    splitting_coord_.resize(sz);
    splitting_val_.resize(sz);

    for(int i = 0; i < nb_points; i++) {
        point_index_[i] = i;
    }

    create_kd_tree_recursive(1, 0, nb_points);

    // Compute the bounding box.
    for(int c = 0; c < dimension(); ++c) {
        bbox_min_[c] = 1e30;
        bbox_max_[c] = -1e30;
    }

    for(int i = 0; i < nb_points; ++i) {
        const float* p = point_ptr(i);
        for(int c = 0; c < dimension(); ++c) {
            bbox_min_[c] = std::min(bbox_min_[c], p[c]);
            bbox_max_[c] = std::max(bbox_max_[c], p[c]);
        }
    }

}

void KdTree::set_points(int nb_points, const float* points) {
    set_points(nb_points, points, dimension());
}

int KdTree::split_kd_node(int node_index, int b, int e) {
    assert(e > b);
    // Do not split leafs
    if(b + 1 == e) {
        return b;
    }

    int splitting_coord = best_splitting_coord(b, e);
    int m = b + (e - b) / 2;
    assert(m < e);

    // sorts the indices in such a way that points's
    // coordinates splitting_coord in [b,m) are smaller
    // than m's and points in [m,e) are
    // greater or equal to m's
    std::nth_element(
            point_index_.begin() + std::ptrdiff_t(b),
            point_index_.begin() + std::ptrdiff_t(m),
            point_index_.begin() + std::ptrdiff_t(e),
            ComparePointCoord(
                nb_points_, points_, stride_, splitting_coord
                )
            );

    // Initialize node's variables (splitting coord and
    // splitting value)
    splitting_coord_[node_index] = splitting_coord;
    splitting_val_[node_index] =
        point_ptr(point_index_[m])[splitting_coord];
    return m;
}

int KdTree::best_splitting_coord(int b, int e) {
    // Returns the coordinates that maximizes
    // point's spread. We should probably
    // use a tradeoff between spread and
    // bbox shape ratio, as done in ANN, but
    // this simple method seems to give good
    // results in our case.
    int result = 0;
    float max_spread = spread(b, e, 0);
    for(int c = 1; c < dimension(); ++c) {
        float coord_spread = spread(b, e, c);
        if(coord_spread > max_spread) {
            result = c;
            max_spread = coord_spread;
        }
    }
    return result;
}

void KdTree::get_nearest_neighbors(int nb_neighbors, const float* query_point, int* neighbors, float* neighbors_sq_dist) const {

    assert(nb_neighbors <= nb_points());

    // Compute distance between query point and global bounding box
    // and copy global bounding box to local variables (bbox_min, bbox_max),
    // allocated on the stack. bbox_min and bbox_max are updated during the
    // traversal of the KdTree (see get_nearest_neighbors_recursive()). They
    // are necessary to compute the distance between the query point and the
    // bbox of the current node.
    float box_dist = 0.0;
    float* bbox_min = (float*) (alloca(dimension() * sizeof(float)));
    float* bbox_max = (float*) (alloca(dimension() * sizeof(float)));
    for(int c = 0; c < dimension(); ++c) {
        bbox_min[c] = bbox_min_[c];
        bbox_max[c] = bbox_max_[c];
        if(query_point[c] < bbox_min_[c]) {
            box_dist += (bbox_min_[c] - query_point[c])*(bbox_min_[c] - query_point[c]);
        } else if(query_point[c] > bbox_max_[c]) {
            box_dist += (bbox_max_[c] - query_point[c])*(bbox_max_[c] - query_point[c]);
        }
    }
    NearestNeighbors NN(nb_neighbors, neighbors, neighbors_sq_dist);
    get_nearest_neighbors_recursive(1, 0, nb_points(), bbox_min, bbox_max, box_dist, query_point, NN);
}


void KdTree::get_nearest_neighbors(int nb_neighbors, int q_index, int* neighbors, float* neighbors_sq_dist) const {
    // TODO: optimized version that uses the fact that
    // we know that query_point is in the search data
    // structure already.
    // (I tryed something already, see in the Attic, 
    //  but it did not give any significant speedup).
    get_nearest_neighbors(
            nb_neighbors, point_ptr(q_index), 
            neighbors, neighbors_sq_dist
            );
}

void KdTree::get_nearest_neighbors_recursive(
        int node_index, int b, int e,
        float* bbox_min, float* bbox_max, float box_dist,
        const float* query_point, NearestNeighbors& NN
        ) const {
    assert(e > b);

    // Simple case (node is a leaf)
    if((e - b) <= MAX_LEAF_SIZE) {
        for(int i = b; i < e; ++i) {
            int p = point_index_[i];
            float d2 = distance2(
                    query_point, point_ptr(p), dimension()
                    );
            NN.insert(p, d2);
        }
        return;
    }

    int coord = splitting_coord_[node_index];
    float val = splitting_val_[node_index];
    float cut_diff = query_point[coord] - val;
    int m = b + (e - b) / 2;

    // If the query point is on the left side
    if(cut_diff < 0.0) {

        // Traverse left subtree
        {
            float bbox_max_save = bbox_max[coord];
            bbox_max[coord] = val;
            get_nearest_neighbors_recursive(
                    2 * node_index, b, m, 
                    bbox_min, bbox_max, box_dist, query_point, NN
                    );
            bbox_max[coord] = bbox_max_save;
        }

        // Update bbox distance (now measures the
        // distance to the bbox of the right subtree)
        float box_diff = bbox_min[coord] - query_point[coord];
        if(box_diff > 0.0) {
            box_dist -= box_diff*box_diff;
        }
        box_dist += cut_diff*cut_diff;

        // Traverse the right subtree, only if bbox
        // distance is nearer than furthest neighbor,
        // else there is no chance that the right
        // subtree contains points that will change
        // anything in the nearest neighbors NN.
        if(box_dist <= NN.furthest_neighbor_sq_dist()) {
            float bbox_min_save = bbox_min[coord];
            bbox_min[coord] = val;
            get_nearest_neighbors_recursive(
                    2 * node_index + 1, m, e, 
                    bbox_min, bbox_max, box_dist, query_point, NN
                    );
            bbox_min[coord] = bbox_min_save;
        }
    } else {
        // else the query point is on the right side
        // (then do the same with left and right subtree
        //  permutted).
        {
            float bbox_min_save = bbox_min[coord];
            bbox_min[coord] = val;
            get_nearest_neighbors_recursive(
                    2 * node_index + 1, m, e, 
                    bbox_min, bbox_max, box_dist, query_point, NN
                    );
            bbox_min[coord] = bbox_min_save;
        }

        // Update bbox distance (now measures the
        // distance to the bbox of the left subtree)
        float box_diff = query_point[coord] - bbox_max[coord];
        if(box_diff > 0.0) {
            box_dist -= box_diff*box_diff;
        }
        box_dist += cut_diff*cut_diff;

        if(box_dist <= NN.furthest_neighbor_sq_dist()) {
            float bbox_max_save = bbox_max[coord];
            bbox_max[coord] = val;
            get_nearest_neighbors_recursive(
                    2 * node_index, b, m, 
                    bbox_min, bbox_max, box_dist, query_point, NN
                    );
            bbox_max[coord] = bbox_max_save;
        }
    }
}

