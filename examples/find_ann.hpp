#ifndef _find_ann_hpp
#define _find_ann_hpp

#include <iostream>  // Input/Output streams
#include <vector>    // STD Dynamic vectors
#include <fstream>   // Open file
#include <sstream>   // Open file
#include <cmath>     // Common math, pow
#include <algorithm> // sort
#include <numeric>   // std::iota
#include <random>
#include <string>
#include <chrono>
#include <ctime>

using namespace std;

//------STANDARD VECTOR ALGEBRA-----------------
inline double dist(double* x, double* y, int d) 
{
  double k = 0.;
   for (int i = 0; i < d; i++) k += pow(x[i] - y[i], 2.);
   return sqrt(k);
 }

inline double norm(double* x, int d) 
{
  double k = 0.;
  for (int i = 0; i < d; i++) k += pow(x[i], 2.);
  return sqrt(k);
}

inline double dot_product(double* x, double* y, int d) 
{
  double k = 0.;
  for (int i = 0; i < d; i++) k += x[i]*y[i];
  return k;
}

//--------------DISTANCE MATRIX------------------
// finds distances between all data points with indices from index_subset
void find_distance_matrix(vector<double> &data, int d, vector<int> &index_subset, vector<vector<double>> &distances)
{
    int subset_size = index_subset.size();
    distances.resize(subset_size, vector<double>(subset_size, 0.0));
        
    for (int i = 0; i < subset_size; i++) 
    {
        for (int j = i + 1; j < subset_size; j++) 
        {
            distances[j][i] = dist(&data[index_subset[i] * d], &data[index_subset[j] * d], d);
            distances[i][j] = distances[j][i];
        }
    } 
}

//--------------READ DATA FROM FILE------------------

void read_from_file(string filename, vector<double> &data)
{
    {
    ifstream f(filename);
    string l;
    while (getline(f, l))
    {
      istringstream sl(l);
      string s;
      while (getline(sl, s, ','))
        data.push_back(stod(s));
    }
    f.close();
  }
}

// for margin in rprojection tree construction, not used so far
// double estimate_set_diameter(vector<double> &data, int n, int d, int* cur_indices, int cur_node_size, mt19937 &generator)
// {
//      uniform_int_distribution<int> uniform_index(0, cur_node_size - 1);
//      int first_index = uniform_index(generator);
//      int second_index = 0;
//      double max_dist = -1;
//      double cur_dist = -1;
//     for (int i = 0; i < cur_node_size; i++) {
//         double cur_dist = dist(&data[cur_indices[i] * d], &data[cur_indices[first_index] * d], d);
//         if (cur_dist > max_dist) {
//             max_dist = cur_dist;
//             second_index = i;
//         }
//     }
//     return max_dist;
// }

//----------------------FIND APPROXIMATE NEAREST NEIGHBORS FROM PROJECTION TREE---------------------------

// 1. CONSTRUCT THE TREE
// leaves - list of all indices such that leaves[leaf_sizes[i]]...leaves[leaf_sizes[i+1]] 
// belong to i-th leaf of the projection tree
// gauss_id and gaussian samples - for the fixed samples option 
void construct_projection_tree(vector<double> &data, int n, int d, int min_leaf_size, 
                               vector<int> &cur_indices, int start, int cur_node_size,
                               vector<int> &leaves, vector<int> &leaf_sizes, mt19937 &generator);
//                               int &gauss_id, const vector<double> &gaussian_samples)
// {
//     if (cur_node_size < min_leaf_size) 
//     {
//         int prev_size = leaf_sizes.back();
//         leaf_sizes.push_back(cur_node_size + prev_size);  
//         for (int i = 0; i < cur_node_size; i++)
//         {
//             leaves.push_back(cur_indices[start + i]);
//         }
//         return;
//     }

//     // choose random direction
//     vector<double> direction_vector(d);
//     normal_distribution<double> normal_distr(0.0,1.0);
//     for (int i = 0; i < d; i++) 
//     {
//         direction_vector[i] = normal_distr(generator);
//         //  option for fixed direction samples
//         //  direction_vector[i] = gaussian_samples[gauss_id*d + i];
//         //  cout << direction_vector[i] << ' ';
//     }
//     // gauss_id++;
//     double dir_vector_norm = norm(&direction_vector[0], d);
//     for (int i = 0; i < d; i++) 
//     {
//         direction_vector[i] /= dir_vector_norm;
//     }

//     // choose margin (MARGIN IS CURRENTLY NOT USED)
//     //double diameter = estimate_set_diameter(data, n, d, cur_indices, cur_node_size, generator);
//     //uniform_real_distribution<double> uniform_on_segment(-1, 1);
//     //double delta = uniform_on_segment(generator);
//     //double margin = delta*6*diameter/sqrt(d);


//     // find relative coordinates
//     vector<double> relative_coordinates(cur_node_size, 0.0);
//     for (int i = 0; i < cur_node_size; i++)
//     {
//         relative_coordinates[i] = dot_product(&data[cur_indices[start + i] * d], &direction_vector[0], d);
//     }
  
//     // median split
//     vector<int> idx(cur_node_size);
//     iota(idx.begin(), idx.end(), 0);
//     std::sort(idx.begin(), idx.end(), [&](int a, int b) {
//         return relative_coordinates[a] < relative_coordinates[b];   
//     });
//     vector<int> cur_indices_sorted(cur_node_size, 0);
//     for (int i = 0; i < cur_node_size; i++)
//     {
//       cur_indices_sorted[i] = cur_indices[start + idx[i]];
//     }
//     for (int i = start; i < start + cur_node_size; i++)
//     {
//       cur_indices[i] = cur_indices_sorted[i - start];
//     }
    
//     int half_size =  (int)cur_node_size / 2;


//     construct_projection_tree(data, n, d, min_leaf_size, 
//                                cur_indices, start, half_size, 
//                                leaves, leaf_sizes, generator);
//                                //, gauss_id, gaussian_samples);

//     construct_projection_tree(data, n, d, min_leaf_size, 
//                                cur_indices, start + half_size, cur_node_size - half_size, 
//                                leaves, leaf_sizes, generator);
//                                //, gauss_id, gaussian_samples);

// }

// 2. FIND CLOSEST POINTS INSIDE LEAVES
// find ann_number exact neighbors for every point among the points within its leaf (in randomized projection tree)
void find_neibs_in_tree(vector<double> &data, int n, int d, int ann_number, vector<int> &leaves, 
                               vector<int> &leaf_sizes, vector<int> &neighbors, vector<double> &neighbor_scores);
// {
//     for (int leaf = 0; leaf < leaf_sizes.size() - 1; leaf++) 
//     {
//         // initialize size and content of the current leaf
//         int cur_leaf_size = leaf_sizes[leaf+1] - leaf_sizes[leaf];
//         vector<int> index_subset(cur_leaf_size, 0); // list of indices in the current leaf
//         for (int i = 0; i < index_subset.size(); i++)
//         {
//             index_subset[i] = leaves[leaf_sizes[leaf] + i];
//         }

//         vector<vector<double>> leaf_dists;
//         find_distance_matrix(data, d, index_subset, leaf_dists);
        
//         // record ann_number closest points in each leaf to neighbors
//         for (int i = 0; i < cur_leaf_size; i++) 
//         {
//             vector<int> idx(cur_leaf_size);
//             iota(idx.begin(), idx.end(), 0);
//             std::sort(idx.begin(), idx.end(), [&](int i1, int i2) {
//                 return leaf_dists[i][i1] < leaf_dists[i][i2];
//              });
//             for (int j = 0; j < ann_number; j++)
//             {
//                 int neibid = leaves[leaf_sizes[leaf] + idx[j]];
//                 double neibscore = leaf_dists[i][idx[j]];
//                 neighbors[index_subset[i] * ann_number + j] = neibid;
//                 neighbor_scores[index_subset[i] * ann_number + j] = neibscore;
//             }
//         }
//     }
// }

// 3. FIND ANN IN ONE TREE SAMPLE
void find_ann_candidates(vector<double> &data, int n, int d, int ann_number, 
                         vector<int> &neighbors, vector<double> &neighbor_scores);
                         //vector<double> &gaussian_samples)
// {
//     random_device rd;
//     mt19937 generator(rd());

//     int min_leaf_size = 6*ann_number;
    
//     vector<int> leaves;
//     vector<int> leaf_sizes;
//     leaf_sizes.push_back(0);

//     int cur_node_size = n;
//     int start = 0;
//    //  int gauss_id = 0;
//     vector<int> cur_indices(cur_node_size);
//     iota(cur_indices.begin(), cur_indices.end(), 0);

//      construct_projection_tree(data, n, d, min_leaf_size, 
//                                cur_indices, start, cur_node_size, 
//                                leaves, leaf_sizes, generator);
//                                // , gauss_id, gaussian_samples);

//     find_neibs_in_tree(data, n, d, ann_number, leaves, leaf_sizes, neighbors, neighbor_scores);
// }

//---------------CHOOSE BEST NEIGHBORS FROM TWO TREE SAMPLES------------------

// take closest neighbors from neighbors and new_neighbors, write them to neighbors and their scores to neighbor_scores
void choose_best_neighbors(vector<int> &neighbors, vector<double> &neighbor_scores, 
                           vector<int> &new_neighbors, vector<double> &new_neighbor_scores, int ann_number);
// {
//     for (int vertex = 0; vertex < neighbors.size(); vertex = vertex + ann_number)
//     {
//         vector<int> cur_neighbors(ann_number, 0);
//         vector<double> cur_neighbor_scores(ann_number, 0.0);
//         int iter1 = 0;
//         int iter2 = 0;
//         int cur = 0;
//         while ((iter1 < ann_number) && (iter2 < ann_number) && (cur < ann_number))
//         {
//             if (neighbor_scores[vertex+iter1] > new_neighbor_scores[vertex+iter2])
//             {
//                 cur_neighbors[cur] = new_neighbors[vertex+iter2];
//                 cur_neighbor_scores[cur] = new_neighbor_scores[vertex+iter2];
//                 iter2++;
//             } else {
//                 cur_neighbors[cur] = neighbors[vertex+iter1];
//                 cur_neighbor_scores[cur] = neighbor_scores[vertex+iter1];
//                 if (neighbors[vertex+iter1] == new_neighbors[vertex+iter2])
//                 {
//                     iter2++;
//                 }
//                 iter1++;
//             }
//             cur++;
//         }
//         while (cur < ann_number)
//         {
//             if (iter1 == ann_number)
//             {
//                 cur_neighbors[cur] = new_neighbors[vertex+iter2];
//                 cur_neighbor_scores[cur] = new_neighbor_scores[vertex+iter2];
//                 iter2++;
//             }
//             else 
//             {
//                 cur_neighbors[cur] = neighbors[vertex+iter1];
//                 cur_neighbor_scores[cur] = neighbor_scores[vertex+iter1];
//                 iter1++;
//             }
//             cur++;
//         }

//         for(int i = 0; i < ann_number; i++)
//         {
//             neighbors[vertex+i] = cur_neighbors[i];
//             neighbor_scores[vertex+i] = cur_neighbor_scores[i];
//         }
//     }

// }

//----------------QUALITY CHECK WITH TRUE NEIGHBORS----------------------------
void find_true_nn(vector<double> &data, int n, int d, int ann_number, vector<int> &n_neighbors, vector<double> &n_neighbor_scores);
// {
//          // create full distance matrix
//         vector<vector<double>> all_dists;
//         vector<int> all_ids(n); // index subset = everything
//         iota(all_ids.begin(), all_ids.end(), 0);
//         find_distance_matrix(data, d, all_ids, all_dists);

//        // record ann_number closest points in each leaf to neighbors
//         for (int i = 0; i < n; i++) 
//         {
//             vector<int> idx(n);
//             iota(idx.begin(), idx.end(), 0);
//             std::sort(idx.begin(), idx.end(), [&](int i1, int i2) {
//                 return all_dists[i][i1] < all_dists[i][i2];
//              });
//             for (int j = 0; j < ann_number; j++)
//             {
//                 n_neighbors[i * ann_number + j] = idx[j];
//                 n_neighbor_scores[i * ann_number + j] = all_dists[i][idx[j]];
//             }
//         }
// }

// quality = average fraction of ann_number approximate neighbors (neighbors), which are 
// within the closest ann_number of true neighbors (n_neighbors); average is taken over all data points
double check_quality(vector<double> &data, int n, int d, int ann_number, vector<int> &neighbors);
// {
//     vector<int> n_neighbors(n*ann_number, 0);
//     vector<double> n_neighbor_scores(n*ann_number, 0.0);
//     auto start_nn = chrono::system_clock::now();
//     find_true_nn(data, n, d, ann_number, n_neighbors, n_neighbor_scores);
//     auto end_nn = chrono::system_clock::now();
//     chrono::duration<double> elapsed_seconds_nn = end_nn-start_nn;
//     cout << "elapsed time for exact neighbor search: " << elapsed_seconds_nn.count() << " sec" << endl;

//     vector<double> quality_vec;
//     for (int i = 0; i < n; i++) 
//     {
//         int iter1 = ann_number*i;
//         int iter2 = ann_number*i;
//         int num_nei_found = 0;
//         while (iter2 < ann_number*(i+1))
//         {
//             if (neighbors[iter1] == n_neighbors[iter2])
//             {
//                 iter1++;
//                 iter2++;
//                 num_nei_found++;
//             }
//             else
//             {
//                 iter2++;
//             }
//         }
//        quality_vec.push_back((double)num_nei_found/ann_number);
//     }
//     cout << endl;
  
//     double ann_quality = 0.0;
//     for (int i = 0; i < quality_vec.size(); i++)
//     {
//         ann_quality += quality_vec[i];
//     }
//     return (double)ann_quality/quality_vec.size();
// }

//------------ITERATE OVER SEVERAL PROJECTION TREES TO FIND ANN----------------
void find_approximate_neighbors(vector<double> &data, int n, int d, 
                                    int num_iters, int ann_number, 
                                    vector<int> &neighbors, vector<double> &neighbor_scores);
                                    // vector<double> &gaussian_samples
// {
//     find_ann_candidates(data, n, d, ann_number, neighbors, neighbor_scores);
//                         //, gaussian_samples);
//     for (int iter = 1; iter < num_iters; iter++)
//     {
//         vector<int> new_neighbors(n*ann_number, 0);
//         vector<double> new_neighbor_scores(n*ann_number, 0.0);   
//         find_ann_candidates(data, n, d, ann_number, new_neighbors, new_neighbor_scores);
//         choose_best_neighbors(neighbors, neighbor_scores, new_neighbors, new_neighbor_scores, ann_number);
//         cout << "iter " << iter << " done" << endl;       
//         }

// }


#endif


