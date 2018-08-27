# include <cmath>
# include <iostream>
# include <vector>
using namespace std;

enum class KernelType {
  GAUSSIAN,      /*!< Radial basis function Gaussian kernel */
  LAPLACIAN,     /*!< Radial basis function Laplacian kernel */
  USER_DEFINED   /*!< User defined kernel */
};

enum class ClusteringMethod {
  NATURAL,       /*!< No ordering */
  KMEANS,        /*!< Recursive 2-means*/
  KDTREE,        /*!< Recursive binary space partitioning tree in d dimensions.*/
  PCA,           /*!< Recursive principal component analysis.*/
};

enum class DataSetType {
  TRAINING,       /*!< Training dataset */
  TEST,           /*!< Test dataset */
  VALIDATION      /*!< Validation dataset */
};

class GaussianKernel {

  private:
    double h;
    int d;

  public:
    GaussianKernel(int in_d, double in_h): d(in_d), h(in_h) { }
    
    // Square distance
    inline double dist2(const vector<double> &x, const vector<double> &y) {
      double k = 0.;
      for (int i = 0; i < d; i++)
        k += pow(x[i] - y[i], 2.);
      return k;
    }

    // Distance
    inline double dist(const vector<double> &x, const vector<double> &y) { 
      return sqrt(dist2(x, y));
    }

    // Kernel evaluation
    double operator() (const vector<double> &x, const vector<double> &y) {
      return exp(-dist2(x, y) / (2. * h * h));
    }  

};

class LaplacianKernel {

  private:
    double h;
    int d;
  
  public:
    LaplacianKernel(int d_in, double h_in): d(d_in), h(h_in) { }

    // Distance
    inline double dist(const vector<double> &x, const vector<double> &y) {
      double k = 0.;
      for (int i = 0; i < d; i++)
        k += fabs(x[i] - y[i]);
      return k;
    }

    // Kernel evaluation
    double operator() (const vector<double> &x, const vector<double> &y) {
      return exp(-dist(x, y) / h);
    }

};

class KernelDataset{

  private:
    size_t n;
    size_t d;
    DataSetType set_type;
    vector<double> data;
    vector<int> labels;
    string data_filename;
    string labels_filename;

  public:
    KernelDataset(size_t n_input, int d_input, 
    DataSetType set_type_input, 
    string data_filename_input, string labels_filename_input)
    : n(n_input), d(d_input), set_type(set_type_input),
      data_filename(data_filename_input), 
      labels_filename(labels_filename_input) {

        cout << "Reading data and labels from files" << endl;
     
    }

    void cluster(ClusteringMethod cluster_method){

      switch (cluster_method) {

        case ClusteringMethod::NATURAL: 
          cout << "NATURAL" << endl;
          break;
        
        case ClusteringMethod::KMEANS: 
          cout << "KMEANS" << endl;
          break;
        
        case ClusteringMethod::KDTREE: 
          cout << "KDTREE" << endl;
          break;
        
        case ClusteringMethod::PCA: 
          cout << "PCA" << endl;
          break;
       }
    }

    void compute_ann(int k, DenseM_t& ann, DenseM_t& scores, 
      const elem_t& Aelem, HSSOptions hss_opts){
        cout << "computing ANN" << endl;
    }

};

int main(){
  
  int dim = 2;
  int h   = 1.0;

  vector<double> p1 {1., 1.};
  vector<double> p2 {2., 2.};

  GaussianKernel gKernel(dim, h);
  cout << "GaussianKernel evaluation " << gKernel(p1, p2) << endl;

  // 1. Loading data
  KernelDataset training_data   (1000, 8, DataSetType::TRAINING,   "./data/SUSY10k_train_data.csv", "./data/SUSY10k_train_label.csv");
  KernelDataset testing_data    (1000, 8, DataSetType::TEST,       "./data/SUSY10k_test_data.csv",  "./data/SUSY10k_test_label.csv");
  KernelDataset validation_data (1000, 8, DataSetType::VALIDATION, "./data/SUSY10k_valid_data.csv", "./data/SUSY10k_valid_label.csv");


  HSSOptions<double> hss_opts;
  HSSPartitionTree cluster_tree;

  // 2. Compute cluster tree
  HSSMatrix HSS = HSSMatrix<double>(cluster_tree, hss_opts);

  // 3. Cluster data
  train.cluster(ClusteringMethod::KMEANS, cluster_tree, hss_opts);

  // 4. Compute ANN
  int k = 32,
  DenseM_t<double> ann(n,k);
  DenseM_t<int>    scores(n,k);
  train.compute_ann(ann, scores);

  // Compress with ANN
  HSS.compress_ann(ann, scores, Aelem, opts)

  // Fit: Create matrix factorization with kernel paratemers
  fit(HSS, gKernel, train_data, test_data);

  vector<double> prediction_vector;

  // Test the model accuracy on unseen data
  predict(HSS, validation_data, prediction_vector);

  return 0;
}
