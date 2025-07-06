import ANNet;

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
using namespace std;

static std::vector<double> oneHot(int classIndex, int totalClasses = 5) {
    std::vector<double> label(totalClasses, 0);
    label[classIndex] = 1;
    return label;
}

static std::vector<std::pair<std::vector<double>, std::vector<double>>>
readCSVToDataset(const std::string& filename, int numFeatures, int numClasses) {
    std::ifstream file(filename);
    std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return dataset;
    }

    std::unordered_map<std::string, int> labelMap = {
        {"setosa", 0},
        {"versicolor", 1},
        {"virginica", 2}
    };

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> features;

        for (int i = 0; i < numFeatures; ++i) {
            if (!std::getline(ss, token, ',')) break;
            features.push_back(std::stod(token));
        }

        if (!std::getline(ss, token, ',')) continue;

        if (labelMap.find(token) == labelMap.end()) {
            std::cerr << "Unknown label: " << token << std::endl;
            continue;
        }

        int labelIndex = labelMap[token];
        std::vector<double> oneHotLabel = oneHot(labelIndex, numClasses);

        dataset.push_back({ features, oneHotLabel });
    }

    file.close();
    return dataset;
}

int main() {
    vector<pair<vector<double>, vector<double>>> dataset = readCSVToDataset("data.csv",4,3);
    pair<vector<pair<vector<double>, vector<double>>>, vector<pair<vector<double>, vector<double>>>>
        stratifiedDataset = stratifiedSampling(dataset);

    vector<pair<vector<double>, vector<double>>> trainingDataset = stratifiedDataset.first;
    vector<pair<vector<double>, vector<double>>> testingDataset = stratifiedDataset.second;

    Network model;
    model.InputLayer(dataset);
    model.HiddenLayer(4);
    model.HiddenLayer(4);
    model.OutputLayer(3);

    model.train(0.005, 0.001, 0.5, 200);

    model.test(testingDataset);
    model.evaluation();

    cout << endl;
    model.showMatrix();
    cout << endl;

    vector<double> inputs = { 6.9,3.2,5.5,2.2 };
    int predictedClass = model.predict(inputs);
    cout << endl;
    cout << "Predicted Class : " << predictedClass + 1 << endl;
    cout << endl;

    return 0;
}
