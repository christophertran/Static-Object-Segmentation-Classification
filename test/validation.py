import os
import collections

ground_truth1 = os.path.join("2022-02-17-18-06-21_ground_truth.txt")
results1 = os.path.join("2022-02-17-18-06-21_results.txt")

ground_truth2 = os.path.join("2022-02-17-18-13-23_ground_truth.txt")
results2 = os.path.join("2022-02-17-18-13-23_results.txt")


def total_validate(_ground_truth, _results):
    ground_truth_fs = open(_ground_truth)
    results_fs = open(_results)
    truth_counter = collections.Counter(
        {"Humans": 0, "Traffic Lights": 0, "Street Signs": 0, "Cars": 0}
    )
    result_counter = collections.Counter(
        {"Humans": 0, "Traffic Lights": 0, "Street Signs": 0, "Cars": 0}
    )

    for truth in ground_truth_fs.readlines():
        line = truth.strip().split(":")

        if len(line) > 1 and line[-1]:
            key, value = line
            truth_counter.update([key] * int(value.strip()))

    for result in results_fs.readlines():
        line = result.strip().split(":")

        if len(line) > 1 and line[-1]:
            key, value = line
            result_counter.update([key] * int(value.strip()))

    print("Truth Total Counts")
    truth_total = 0
    for k, v in truth_counter.items():
        print(f"{k}: {v}")
        truth_total += v

    print("\n")

    print("Result Total Counts")
    result_total = 0
    for k, v in result_counter.items():
        print(f"{k}: {v}")
        result_total += v

    print("\nAccuracy:")
    print(
        (result_counter["Humans"] / truth_counter["Humans"])
        if truth_counter["Humans"] > 0
        else 0
    )
    print(
        (result_counter["Traffic Lights"] / truth_counter["Traffic Lights"])
        if truth_counter["Traffic Lights"] > 0
        else 0
    )
    print(
        (result_counter["Street Signs"] / truth_counter["Street Signs"])
        if truth_counter["Street Signs"] > 0
        else 0
    )
    print(
        (result_counter["Cars"] / truth_counter["Cars"])
        if truth_counter["Cars"] > 0
        else 0
    )


def basic_validate(_ground_truth, _results):
    ground_truth_fs = open(_ground_truth)
    results_fs = open(_results)
    counter = collections.Counter()

    for truth, result in zip(ground_truth_fs.readlines(), results_fs.readlines()):
        # counter.update([truth.strip() == result.strip()])
        truth = truth.strip().split(":")
        result = result.strip().split(":")

        if len(truth) > 1 and truth[-1] and len(result) > 1 and result[-1]:
            counter.update([truth == result])

    print("Total Predictions Correctness")
    total = 0
    for k, v in counter.items():
        if k:
            print(f"Correct: {v}")
        else:
            print(f"Incorrect: {v}")

        total += v
    print("Total Predictions:", total)

    print("Accuracy:", counter[True] / total)


def main():
    print("=" * 10)
    basic_validate(ground_truth1, results1)
    print("=" * 10)

    print("=" * 10)
    basic_validate(ground_truth2, results2)
    print("=" * 10)

    # print("=" * 10)
    # total_validate(ground_truth1, results1)
    # print("=" * 10)

    # print("=" * 10)
    # total_validate(ground_truth2, results2)
    # print("=" * 10)


if __name__ == "__main__":
    main()
