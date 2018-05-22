#ifndef KD_TREE
#define KD_TREE

#include <vector>
#include <cassert>
#include <limits>
#include <algorithm>

class KdTree {
    public:

        /**
         * \brief Creates a new KdTree.
         * \param[in] dim dimension of the points
         */
        KdTree(int dim);

        virtual void set_points(int nb_points, const float* points, int stride);

        virtual void set_points(int nb_points, const float* points);

        virtual void get_nearest_neighbors(int nb_neighbors, const float* query_point, int* neighbors, float* neighbors_sq_dist) const;

        virtual void get_nearest_neighbors(int nb_neighbors, int query_point, int* neighbors, float* neighbors_sq_dist) const;


        int dimension() const {
            return dimension_;
        }

        int nb_points() const {
            return nb_points_;
        }

        const float* point_ptr(int i) const {
            assert(i < nb_points());
            return points_ + i * stride_;
        }

        virtual ~KdTree();

    protected:
        static const int MAX_LEAF_SIZE = 16;

        /**
         * \brief Returns the maximum node index in subtree.
         * \param[in] node_id node index of the subtree
         * \param[in] b first index of the points sequence in the subtree
         * \param[in] e one position past the last index of the point
         *  sequence in the subtree
         */
        static int max_node_index(int node_id, int b, int e) {
            if(e - b <= MAX_LEAF_SIZE) {
                return node_id;
            }
            int m = b + (e - b) / 2;
            return std::max(max_node_index(2 * node_id, b, m), max_node_index(2 * node_id + 1, m, e));
        }

        /**
         * \brief The context for traversing a KdTree.
         * \details Stores a sorted sequence of (point,distance)
         *  couples.
         */
        struct NearestNeighbors {

            /**
             * \brief Creates a new NearestNeighbors
             * \details Storage is provided
             * and managed by the caller.
             * Initializes neighbors_sq_dist[0..nb_neigh-1]
             * to Numeric::max_float64() and neighbors[0..nb_neigh-1]
             * to int(-1).
             * \param[in] nb_neighbors_in number of neighbors to retreive
             * \param[in] neighbors_in storage for the neighbors, allocated
             *  and managed by caller
             * \param[in] neighbors_sq_dist_in storage for neighbors squared
             *  distance, allocated and managed by caller
             */
            NearestNeighbors(int nb_neighbors_in, int* neighbors_in, float* neighbors_sq_dist_in) :
                nb_neighbors(nb_neighbors_in), neighbors(neighbors_in), neighbors_sq_dist(neighbors_sq_dist_in) {
                    for(int i = 0; i < nb_neighbors; ++i) {
                        neighbors[i] = int(-1);
                        neighbors_sq_dist[i] = std::numeric_limits<float>::max();
                    }
                }

            /**
             * \brief Gets the squared distance to the furthest
             *  neighbor.
             */
            float furthest_neighbor_sq_dist() const {
                return neighbors_sq_dist[nb_neighbors - 1];
            }

            /**
             * \brief Inserts a new neighbor.
             * \details Only the nb_neighbor nearest points are kept.
             * \param[in] neighbor the index of the point
             * \param[in] sq_dist the squared distance between the point
             *  and the query point.
             */
            void insert(int neighbor, float sq_dist) {
                if(sq_dist >= furthest_neighbor_sq_dist()) {
                    return;
                }
                int i = nb_neighbors;
                while(i != 0 && neighbors_sq_dist[i - 1] > sq_dist) {
                    if(i < nb_neighbors) {
                        neighbors[i] = neighbors[i - 1];
                        neighbors_sq_dist[i] = neighbors_sq_dist[i - 1];
                    }
                    --i;
                }
                assert(i < nb_neighbors);
                neighbors[i] = neighbor;
                neighbors_sq_dist[i] = sq_dist;
            }

            int nb_neighbors;
            int* neighbors;
            float* neighbors_sq_dist;
        };

        /**
         * \brief Computes the coordinate along which a point
         *   sequence will be splitted.
         * \param[in] b first index of the point sequence
         * \param[in] e one position past the last index of the point sequence
         */
        int best_splitting_coord(int b, int e);

        /**
         * \brief Computes the extent of a point sequence along a coordinate.
         * \param[in] b first index of the point sequence
         * \param[in] e one position past the last index of the point sequence
         * \param[in] coord coordinate along which the extent is measured
         */
        float spread(int b, int e, int coord) {
            float minval = std::numeric_limits<float>::max();
            float maxval = -minval;
            for(int i = b; i < e; ++i) {
                float val = point_ptr(point_index_[i])[coord];
                minval = std::min(minval, val);
                maxval = std::max(maxval, val);
            }
            return maxval - minval;
        }

        /**
         * \brief Creates the subtree under a node.
         * \param[in] node_index index of the node that represents
         *  the subtree to create
         * \param[in] b first index of the point sequence in the subtree
         * \param[in] e one position past the last index of the point
         *  index in the subtree
         */
        void create_kd_tree_recursive(int node_index, int b, int e) {
            if(e - b <= MAX_LEAF_SIZE) {
                return;
            }
            int m = split_kd_node(node_index, b, e);
            create_kd_tree_recursive(2 * node_index, b, m);
            create_kd_tree_recursive(2 * node_index + 1, m, e);
        }

        /**
         * \brief Computes and stores the splitting coordinate
         *  and splitting value of the node node_index, that
         *  corresponds to the [b,e) points sequence.
         *
         * \return a node index m. The point sequences
         *  [b,m) and [m,e) correspond to the left
         *  child (2*node_index) and right child (2*node_index+1)
         *  of node_index.
         */
        int split_kd_node(int node_index, int b, int e);

        /**
         * \brief The recursive function to implement KdTree traversal and
         *  nearest neighbors computation.
         * \details Traverses the subtree under the
         *  node_index node that corresponds to the
         *  [b,e) point sequence. Nearest neighbors
         *  are inserted into neighbors during
         *  traversal.
         * \param[in] node_index index of the current node in the Kd tree
         * \param[in] b index of the first point in the subtree under
         *  node \p node_index
         * \param[in] e one position past the index of the last point in the
         *  subtree under node \p node_index
         * \param[in,out] bbox_min coordinates of the lower
         *  corner of the bounding box.
         *  Allocated and managed by caller.
         *  Modified by the function and restored on exit.
         * \param[in,out] bbox_max coordinates of the
         *  upper corner of the bounding box.
         *  Allocated and managed by caller.
         *  Modified by the function and restored on exit.
         * \param[in] bbox_dist squared distance between
         *  the query point and a bounding box of the
         *  [b,e) point sequence. It is used to early
         *  prune traversals that do not generate nearest
         *  neighbors.
         * \param[in] query_point the query point
         * \param[in,out] neighbors the computed nearest neighbors
         */
        void get_nearest_neighbors_recursive(int node_index, int b, int e, float* bbox_min, float* bbox_max, float bbox_dist, const float* query_point, NearestNeighbors& neighbors) const;

    protected:
        std::vector<int> point_index_;
        std::vector<int> splitting_coord_;
        std::vector<float> splitting_val_;
        std::vector<float> bbox_min_;
        std::vector<float> bbox_max_;
        int dimension_;
        int nb_points_;
        int stride_;
        const float* points_;
};

#endif

