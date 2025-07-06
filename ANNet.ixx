export module ANNet;

import <iostream>;
import <vector>;
import <random>;
import <cmath>;
import <stdexcept>;
import <algorithm>;
import <unordered_map>;
import <utility>;
import <string>;

std::default_random_engine GENERATOR;
std::normal_distribution<double> DISTRIBUTION(0.0, 1.0);

static double randomGenerator(int layerSize) {
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / layerSize));
    return dist(GENERATOR);
}

static size_t argmax(std::vector<double> predicted) {
    size_t idx = 0;
    double max = predicted[idx];
    for (size_t i = 1; i < predicted.size(); i++) {
        if (predicted[i] > max) {
            max = predicted[i];
            idx = i;
        }
    }
    return idx;
}

export std::pair<std::vector<std::pair<std::vector<double>, std::vector<double>>>,
    std::vector<std::pair<std::vector<double>, std::vector<double>>>>
    stratifiedSampling(std::vector<std::pair<std::vector<double>, std::vector<double>>>& dataset) {

    std::unordered_map<double, std::vector<std::vector<double>>> classBuckets;
    for (const auto& sample : dataset) {
        std::vector<double> label = sample.second;
        std::vector<double> value = sample.first;
        classBuckets[argmax(label)].push_back(value);
    }

    std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingDataset;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> testingDataset;

    int sizeClassBucketsMap = classBuckets.size();
    for (const auto& bucket : classBuckets) {
        std::vector<std::vector<double>> bucketValues = bucket.second;
        int bucketLabel = bucket.first;

        std::vector<double> temp(sizeClassBucketsMap, 0);
        temp[bucketLabel] = 1;

        size_t size = bucketValues.size();
        size_t count = 0;
        for (const auto& value : bucketValues) {
            std::pair<std::vector<double>, std::vector<double>> append;
            append.first = value;
            append.second = temp;
            if (count < size * 0.8) {
                trainingDataset.push_back(append);
            }
            else {
                testingDataset.push_back(append);
            }
            count++;
        }
    }

    std::shuffle(trainingDataset.begin(), trainingDataset.end(), std::mt19937{ std::random_device{}() });
    std::shuffle(testingDataset.begin(), testingDataset.end(), std::mt19937{ std::random_device{}() });

    std::pair<std::vector<std::pair<std::vector<double>, std::vector<double>>>,
        std::vector<std::pair<std::vector<double>, std::vector<double>>>> outputDatasets;
    outputDatasets.first = trainingDataset;
    outputDatasets.second = testingDataset;

    return outputDatasets;
}

static double CCE(const std::vector<double>& predicted, const std::vector<double>& groundTruth) {
    double loss = 0.0;
    const double epsilon = 1e-15;

    for (size_t i = 0; i < predicted.size(); ++i) {
        double p = std::clamp(predicted[i], epsilon, 1.0 - epsilon);
        loss += -groundTruth[i] * std::log(p);
    }

    return loss / groundTruth.size();
}

static double relu(double x) {
    return (x > 0) ? x : 0;
}

static double derivativeRelu(double x) {
    return (x > 0) ? 1 : 0;
}

class Neuron {
public:
    double BIAS;
    double OUTPUT;
    double NON_ACTIVE_OUTPUT;

    std::vector<double> INPUTS;
    std::vector<double> WEIGHTS;

    Neuron(size_t inputsSize, std::string layerType, int layerSize) {
        this->BIAS = 0;

        for (size_t i = 0; i < inputsSize; i++) {
            double tempWeight = randomGenerator(layerSize);
            this->WEIGHTS.push_back(tempWeight);
        }
    }

    void forward(const std::vector<double>& inputs, std::string layerType) {
        double weightedSum = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            weightedSum += inputs[i] * this->WEIGHTS[i];
        }
        this->INPUTS = inputs;
        this->NON_ACTIVE_OUTPUT = weightedSum + this->BIAS;
        if (layerType == "HiddenLayer") {
            this->OUTPUT = relu(this->NON_ACTIVE_OUTPUT);
        }
    }

    std::vector<double> backwardNeuron(std::vector<double>& prevOutputs, double learningRate, double delta, double dy_dz, std::string layerType, double threshold, double L2_Strength) {        
        double dLoss_dz;

        if (layerType == "OutputLayer") {
            dLoss_dz = delta;
        }
        else {
            dLoss_dz = delta * dy_dz;
        }

        std::vector<double> delta_prev(this->WEIGHTS.size());
        for (size_t i = 0; i < this->WEIGHTS.size(); i++) {
            delta_prev[i] = dLoss_dz * this->WEIGHTS[i];
        }

        std::vector<double> gradients(prevOutputs.size());
        for (size_t i = 0; i < prevOutputs.size(); i++) {
            gradients[i] = dLoss_dz * prevOutputs[i];
        }

        gradients = this->gradientClipping(gradients, threshold);

        for (size_t i = 0; i < prevOutputs.size(); i++) {
            this->WEIGHTS[i] -= learningRate * (gradients[i] + (L2_Strength * this->WEIGHTS[i]));
        }
        this->BIAS -= dLoss_dz * learningRate;

        return delta_prev;
    }

    std::vector<double> gradientClipping(std::vector<double>& gradients, double threshold) {
        double sumSquare = 0;
        for (double g : gradients) {
            sumSquare += std::pow(g, 2);
        }
        double norm = std::sqrt(sumSquare);

        if (norm > threshold) {
            double clip = threshold / norm;
            for (size_t i = 0; i < gradients.size(); i++) {
                gradients[i] = gradients[i] * clip;
            }
        }
        return gradients;
    }
};

class Layer {
public:
    int LAYER_SIZE;
    double LOSS;

    std::string LAYER_TYPE;

    std::vector<double> INPUTS;
    std::vector<Neuron*> LAYER;
    std::vector<double> OUTPUTS;

    Layer(int layerSize, std::vector<double>& inputs, std::string layerType)
        : LAYER_SIZE(layerSize), LAYER_TYPE(layerType) {

        for (int i = 0; i < this->LAYER_SIZE; i++) {
            Neuron* n = new Neuron(inputs.size(), this->LAYER_TYPE, this->LAYER_SIZE);
            this->LAYER.push_back(n);
        }

        std::vector<double> logits;
        for (Neuron* n : this->LAYER) {
            logits.push_back(n->NON_ACTIVE_OUTPUT);
        }

        if (this->LAYER_TYPE == "OutputLayer") {
            std::vector<double> normalized = normalization(logits);
            std::vector<double> softmaxActivatedOutputs = softmax(normalized);

            for (size_t i = 0; i < this->LAYER_SIZE; i++) {
                this->LAYER[i]->OUTPUT = softmaxActivatedOutputs[i];
                this->OUTPUTS.push_back(softmaxActivatedOutputs[i]);
            }
        }
        else {
            for (Neuron* n : this->LAYER) {
                this->OUTPUTS.push_back(n->OUTPUT);
            }
        }
    }

    ~Layer() {
        for (Neuron* neuron : this->LAYER) {
            delete neuron;
        }
    }

    void forward(std::vector<double>& inputs) {
        this->OUTPUTS.clear();

        for (Neuron* n : this->LAYER) {
            n->forward(inputs, this->LAYER_TYPE);
        }

        std::vector<double> logits;
        for (Neuron* n : this->LAYER) {
            logits.push_back(n->NON_ACTIVE_OUTPUT);
        }

        if (this->LAYER_TYPE == "OutputLayer") {
            std::vector<double> normalized = this->normalization(logits);
            std::vector<double> softmaxActivatedOutputs = softmax(normalized);

            for (size_t i = 0; i < this->LAYER_SIZE; i++) {
                this->LAYER[i]->OUTPUT = softmaxActivatedOutputs[i];
                this->OUTPUTS.push_back(softmaxActivatedOutputs[i]);
            }
        }
        else {
            for (Neuron* n : this->LAYER) {
                this->OUTPUTS.push_back(n->OUTPUT);
            }
        }
    }

    std::vector<double> backwardLayer(std::vector<double>& prevOutputs, std::vector<double>& inputDelta, double learningRate, double threshold, double L2_Strength) {
        std::vector<std::vector<double>> deltas;
        int count = 0;

        double dy_dz = 0;

        for (Neuron* n : this->LAYER) {
            if (this->LAYER_TYPE == "OutputLayer") {
                dy_dz = 1.0;
            }
            else {
                dy_dz = derivativeRelu(n->NON_ACTIVE_OUTPUT);
            }

            std::vector<double> delta = n->backwardNeuron(prevOutputs, learningRate, inputDelta[count], dy_dz, this->LAYER_TYPE, threshold, L2_Strength);
            deltas.push_back(delta);
            count++;
        }

        std::vector<double> delta_prev;
        for (size_t i = 0; i < deltas[0].size(); i++) {
            double sum = 0;
            for (size_t j = 0; j < deltas.size(); j++) {
                sum += deltas[j][i];
            }
            delta_prev.push_back(sum);
        }

        return delta_prev;
    }

    std::vector<double> normalization(std::vector<double>& logits) {
        std::vector<double> normalizedLogits;
        normalizedLogits.reserve(logits.size());

        double max = *std::max_element(logits.begin(), logits.end());

        for (double logit : logits) {
            normalizedLogits.push_back(logit - max);
        }

        return normalizedLogits;
    }

    static std::vector<double> softmax(std::vector<double>& outputs) {
        double maxOutput = *std::max_element(outputs.begin(), outputs.end());
        std::vector<double> expOutputs(outputs.size());
        double sumExp = 0.0;

        for (size_t i = 0; i < outputs.size(); ++i) {
            expOutputs[i] = std::exp(outputs[i] - maxOutput);
            sumExp += expOutputs[i];
        }

        double epsilon = 1e-10;
        if (sumExp < epsilon) sumExp = epsilon;

        std::vector<double> probabilities(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            probabilities[i] = expOutputs[i] / sumExp;
        }

        return probabilities;
    }
};

export class Network {
public:
    Network() {}

    ~Network() {
        for (Layer* layer : this->NETWORK) {
            delete layer;
        }
    }

    void InputLayer(std::vector<std::pair<std::vector<double>, std::vector<double>>>& dataset) {
        this->convert(dataset);

        this->INITIAL_INPUTS = this->DATA_INPUTS_VECTOR[0];
        this->CLASS_OUTPUTS = this->DATA_OUTPUTS_VECTOR[0];

        size_t x = this->CLASS_OUTPUTS.size();
        for (size_t i = 0; i < x; i++) {
            this->CONFUSION_MATRIX.push_back(std::vector<int>(x, 0));
        }
    }

    void HiddenLayer(int neuronsCount) {
        std::vector<double> prevLayerOutputs;
        if (this->NETWORK.size() == 0) prevLayerOutputs = this->INITIAL_INPUTS;
        else prevLayerOutputs = this->NETWORK[this->NETWORK.size() - 1]->OUTPUTS;
        this->NETWORK.push_back(new Layer(neuronsCount, prevLayerOutputs, "HiddenLayer"));
    }

    void OutputLayer(int neuronsCount) {
        if (this->NETWORK.size() == 0) throw std::logic_error("Hidden Layer Not Fount");
        std::vector<double> prevLayerOutputs = this->NETWORK[this->NETWORK.size() - 1]->OUTPUTS;
        this->NETWORK.push_back(new Layer(neuronsCount, prevLayerOutputs, "OutputLayer"));
    }

    void train(double learningRate, double L2_Strength, double threshold, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;

            for (size_t sample = 0; sample < this->DATA_INPUTS_VECTOR.size(); ++sample) {
                this->INITIAL_INPUTS = this->DATA_INPUTS_VECTOR[sample];
                this->CLASS_OUTPUTS = this->DATA_OUTPUTS_VECTOR[sample];

                std::vector<std::vector<double>> currentInputsMatrix;
                std::vector<double> currentInput = this->INITIAL_INPUTS;

                // Forward pass
                for (Layer* layer : this->NETWORK) {
                    currentInputsMatrix.push_back(currentInput);
                    layer->forward(currentInput);
                    currentInput = layer->OUTPUTS;
                }

                // Loss
                totalLoss += CCE(currentInput, this->CLASS_OUTPUTS);

                // Backpropagation
                std::vector<double> currentDelta;
                currentDelta.reserve(this->CLASS_OUTPUTS.size());
                for (size_t i = 0; i < this->CLASS_OUTPUTS.size(); i++) {
                    currentDelta.push_back(currentInput[i] - this->CLASS_OUTPUTS[i]);
                }

                for (int i = (int)this->NETWORK.size() - 1; i >= 0; i--) {
                    currentDelta = this->NETWORK[i]->backwardLayer(currentInputsMatrix[i], currentDelta, learningRate, threshold, L2_Strength);
                }
            }

             double avgLoss = totalLoss / this->DATA_INPUTS_VECTOR.size();
             std::cout << "Epoch " << epoch + 1 << " - Loss: " << avgLoss << std::endl;
        }
    }

    void test(std::vector<std::pair<std::vector<double>, std::vector<double>>>& dataset) {
        this->convert(dataset);
        size_t size = this->DATA_INPUTS_VECTOR.size();

        for (size_t i = 0; i < size; i++) {
            this->INITIAL_INPUTS = this->DATA_INPUTS_VECTOR[i];
            this->CLASS_OUTPUTS = this->DATA_OUTPUTS_VECTOR[i];

            std::vector<double> currentInput = this->INITIAL_INPUTS;
            for (Layer* layer : this->NETWORK) {
                layer->forward(currentInput);
                currentInput = layer->OUTPUTS;
            }

            size_t x = argmax(currentInput);
            size_t y = argmax(this->CLASS_OUTPUTS);

            this->CONFUSION_MATRIX[y][x] += 1;
        }
    }

    int predict(std::vector<double>& inputs) {
        std::vector<double> currentInput = inputs;
        for (Layer* layer : this->NETWORK) {
            layer->forward(currentInput);
            currentInput = layer->OUTPUTS;
        }
        return argmax(currentInput);
    }

    void evaluation() {
        double accuracy = this->accuracy();
        std::vector<double> precision = this->precision();
        std::vector<double> recall = this->recall();
        std::vector<double> F1_score = this->f1_Score();

        std::cout << "accuracy : " << accuracy << std::endl;
        for (int i = 0; i < this->DATA_OUTPUTS_VECTOR[0].size(); i++) {
            std::cout << "________" << "Evaluation for class : " << i + 1 << "________" << std::endl;
            std::cout << "precision : " << precision[i] << std::endl;
            std::cout << "recall : " << recall[i] << std::endl;
            std::cout << "F1 Score : " << F1_score[i] << std::endl;
            std::cout << " ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____" << std::endl;
        }
    }

    double accuracy() {
        double acc = 0;
        double den = 0;
        for (size_t i = 0; i < this->CONFUSION_MATRIX.size(); i++) {
            acc += this->truePostiveClass_K(i);
            for (size_t j = 0; j < this->CLASS_OUTPUTS.size(); j++) {
                den += this->CONFUSION_MATRIX[i][j];
            }
        }
        this->ACCURACY = acc / den;
        return this->ACCURACY;
    }

    std::vector<double> precision() {
        std::vector<double> classWisePrecision;
        size_t size = this->CONFUSION_MATRIX.size();
        classWisePrecision.reserve(size);
        for (size_t i = 0; i < size; i++) {
            double TP = truePostiveClass_K(i);
            double FP = falsePositiveClass_K(i);
            if (TP + FP == 0) {
                classWisePrecision.push_back(0);
            }
            else {
                classWisePrecision.push_back(TP / (TP + FP));
            }
        }
        this->CLASS_WISE_PRECISION = classWisePrecision;
        return classWisePrecision;
    }

    std::vector<double> recall() {
        std::vector<double> classWiseRecall;
        size_t size = this->CONFUSION_MATRIX.size();
        classWiseRecall.reserve(size);
        for (size_t i = 0; i < size; i++) {
            double TP = truePostiveClass_K(i);
            double FN = falseNegativeClass_K(i);
            if (TP + FN == 0) {
                classWiseRecall.push_back(0);
            }
            else {
                classWiseRecall.push_back(TP / (TP + FN));
            }
        }
        this->CLASS_WISE_RECALL = classWiseRecall;
        return classWiseRecall;
    }

    std::vector<double> f1_Score() {
        std::vector<double> classWisef1score;
        size_t size = this->CONFUSION_MATRIX.size();
        classWisef1score.reserve(size);
        if (this->CLASS_WISE_PRECISION.empty()) this->precision();
        if (this->CLASS_WISE_RECALL.empty()) this->recall();
        for (size_t i = 0; i < size; i++) {
            double P = this->CLASS_WISE_PRECISION[i];
            double R = this->CLASS_WISE_RECALL[i];
            if (P + R == 0) {
                classWisef1score.push_back(0);
            }
            else {
                classWisef1score.push_back(2 * ((P * R) / (P + R)));
            }
        }
        this->CLASS_WISE_F1_SCORE = classWisef1score;
        return classWisef1score;
    }

    void showMatrix() {
        size_t x = this->CONFUSION_MATRIX.size();
        for (size_t i = 0; i < x; i++) {
            for (size_t j = 0; j < x; j++) {
                std::cout << this->CONFUSION_MATRIX[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    std::vector<Layer*> NETWORK;
    std::vector<double> INITIAL_INPUTS;
    std::vector<double> CLASS_OUTPUTS;

    std::vector<std::vector<double>> DATA_INPUTS_VECTOR;
    std::vector<std::vector<double>> DATA_OUTPUTS_VECTOR;

    std::vector<std::vector<int>> CONFUSION_MATRIX;
    double ACCURACY = 0.0;
    std::vector<double> CLASS_WISE_PRECISION;
    std::vector<double> CLASS_WISE_RECALL;
    std::vector<double> CLASS_WISE_F1_SCORE;

    int truePostiveClass_K(size_t idx) {
        return this->CONFUSION_MATRIX[idx][idx];
    }

    void convert(std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset) {
        this->DATA_INPUTS_VECTOR.clear();
        this->DATA_OUTPUTS_VECTOR.clear();
        for (const auto& sample : dataset) {
            this->DATA_INPUTS_VECTOR.push_back(sample.first);
            this->DATA_OUTPUTS_VECTOR.push_back(sample.second);
        }
    }

    int falsePositiveClass_K(size_t idx) {
        int FP = 0;
        for (size_t i = 0; i < this->CONFUSION_MATRIX.size(); i++) {
            if (idx == i) {
                continue;
            }
            FP += this->CONFUSION_MATRIX[i][idx];
        }
        return FP;
    }

    int falseNegativeClass_K(size_t idx) {
        int FN = 0;
        for (size_t j = 0; j < this->CONFUSION_MATRIX.size(); j++) {
            if (idx == j) {
                continue;
            }
            FN += this->CONFUSION_MATRIX[idx][j];
        }
        return FN;
    }

    int trueNegativeClass_K(size_t idx) {
        int TN = 0;
        for (size_t i = 0; i < this->CONFUSION_MATRIX.size(); i++) {
            if (idx == i) {
                continue;
            }
            for (size_t j = 0; j < this->CONFUSION_MATRIX.size(); j++) {
                if (idx == j) {
                    continue;
                }
                TN += this->CONFUSION_MATRIX[i][j];
            }
        }
        return TN;
    }
};