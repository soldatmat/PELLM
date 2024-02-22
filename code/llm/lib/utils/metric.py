import torch


def get_outputs(model, inputs):
    outputs = torch.empty(len(inputs), dtype=model(inputs[0]).dtype)
    for i in range(len(inputs)):
        outputs[i] = model(inputs[i])
    return outputs


def pair_ranking_accuracy(outputs, labels):
    assert len(outputs) == len(labels)
    correct_pairs = 0
    for i in range(len(outputs)):
        for j in range(len(outputs)):
            true_ranking = labels[i] > labels[j]
            prediction = outputs[i] > outputs[j]
            if true_ranking == prediction:
                correct_pairs = correct_pairs + 1
    return correct_pairs / (len(outputs) * len(outputs))


def treshold_discrimination(outputs, labels, threshold):
    correct = 0
    for i in range(len(outputs)):
        true_disc = labels[i] >= threshold
        prediction = outputs[i] >= threshold
        if true_disc == prediction:
            correct = correct + 1
    return correct / len(outputs)
