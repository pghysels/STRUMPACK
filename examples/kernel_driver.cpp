# include <cmath>
# include <iostream>
# include <vector>
using namespace std;

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
    // KernelDataset& data_set; // Interface v2

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

    // Kernel evaluation (use a pointer)
    double operator() (const vector<double> &x, const vector<double> &y) {
      return exp(-dist2(x, y) / (2. * h * h));
    }  


};

class LaplacianKernel {

  private:
    double h;
  
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

class KernelDataset{ //input data from csv (excel) files

  private:
    size_t n;  // Size of dataset
    size_t d;  // Dimension of each sample
    size_t c;  // Number of classes
    
    DataSetType set_type;

    Dense_t data(n,d);  // [n,d] actual dataset
    vector<int> labels; // [n,1] integer that specifies the class type <0,1,2,...>

    string data_filename;
    string labels_filename;

  public:
    // Constructor
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

    void compute_ann(int k, int iterations, 
                    DenseM_t& ann, DenseM_t& scores, 
                    HSSOptions hss_opts){
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

  // 2. Cluster data
  // cluster_tree is an ouput, by reference.
  training_data.cluster(ClusteringMethod::KMEANS, cluster_tree, hss_opts);

  // 3. HSS instace matrix based on cluster tree
  HSSMatrix HSS = HSSMatrix<double>(cluster_tree, hss_opts);

  // 4. Compute ANN
  int k = 32;   // number of neighbors
  int ann_iter; // number of ANN iterations
  DenseM_t<int>       ann(n,k);    // indices
  DenseM_t<real_t>    scores(n,k); // distances
  training_data.compute_ann(k, ann_iter, ann, ann_iter, scores);


  // Compress with ANN
  HSS.compress_ann(ann, scores, Aelem, opts);
  // HSS.compress_ann(training_data, gKernel, opts);
  // HSS.compress_ann(gKernel, opts);


  // PREDICT: Create matrix factorization with kernel paratemers 
  // and report prediction accuracy on testing data

  vector<n> prediction_vector; // probabilities
  binary_predict(HSS, testing_data, prediction_vector);

  Dense_t<n,c> prediction_vector; // probabilities
  multiclass_predict(HSS, testing_data, prediction_vector);

  return 0;
}



////////////////////////////////////////////////
// Driver1. Expert driver with HSS for PREDICT
////////////////////////////////////////////////
1. Load data (training, testing)
  1.2 Cluster data

2. Instanciate Kernel

3. Create HSS matrix <--- type of matrix

4. Compress
  4.1 Compute ANN

5. Predict (binary or multiclass)


//////////////////////////////////////////////// 
// Driver2. Expert driver with HSS for FIT
////////////////////////////////////////////////

1. Load data(training, validation)
  1.2 Cluster data

2. Instanciate Kernel

3. Create HSS matrix

4. Compress
  4.1 Compute ANN

5. Fit
  5.1 <OpenTuner,GridSearch,Wissam>. INPUT:h, lambda. OUTPUT:pred_acc



////////////////////////////////////////////////
// Driver3. Simple driver.
////////////////////////////////////////////////

1. Get kernel parameters (h), training and validation data
2. Predict, output labels

