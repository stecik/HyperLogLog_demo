import mmh3
from math import sqrt, log
from itertools import product
from random import randint
import pickle
import os
import time


class HyperLogLog:
    def __init__(self, b=14, M=[], S=dict()) -> None:
        self.M = M
        self.b = b
        self.m = 2**self.b
        if not S:
            self.S = self.init_buckets()
        else:
            self.S = S
        self.alpha = self.set_alpha(self.m)
        self.hash_bits = 64

    def set_alpha(self, m):
        # Sets alpha for different values of m
        if m <= 16:
            return 0.673
        elif m <= 32:
            return 0.697
        elif m <= 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    def init_buckets(self):
        return {x: 0 for x in range(self.m)}

    def count_trail_zeros(self, h):
        count = 1
        while not h & 1:
            h >>= 1
            count += 1
            if count == 64:
                return 65
        return count

    def get_first_bits(self, h):
        h >>= self.hash_bits - self.b
        return h

    def compute_buckets(self):
        for a in self.M:
            h = mmh3.hash64(str(a), signed=False)[0]
            p = self.count_trail_zeros(h)
            bucket = self.get_first_bits(h)
            if p > self.S[bucket]:
                self.S[bucket] = p

    def compute_E(self):
        sum_ = sum([2 ** (-1 * item) for item in self.S.values()])
        harmonic_avg = self.m / sum_
        E = self.alpha * self.m * harmonic_avg
        return E

    def num_registers_zero(self):
        count = 0
        for value in self.S.values():
            if value == 0:
                count += 1
        return count

    def correct_E(self, E):
        E_final = E
        # Computes corrected estimate
        # Small range correction --> linear counting
        if E <= (5 * self.m) / 2:
            V = (
                self.num_registers_zero()
            )  # Let V be the number of registers equal to 0.
            if V != 0:
                E_final = self.m * log(self.m / V)

        # Return corrected estimate with relative error of +-1.04/sqrt(m)
        return E_final

    def estimate_cardinality(self):
        self.compute_buckets()
        E = self.compute_E()
        return self.correct_E(E)

    def save(self, name):
        pickle.dump(self.S, open(f"{name}.pickle", "wb"))


def merge_HLL(list_of_HLLs):
    merged_S = dict()
    for HLL in list_of_HLLs:
        for key, value in HLL.S.items():
            if key not in merged_S:
                merged_S[key] = value
            else:
                merged_S[key] = max(merged_S[key], value)
    return merged_S


def estimate_cardinality_merged(merged_S):
    HLL = HyperLogLog(S=merged_S)
    E = HLL.compute_E()
    return HLL.correct_E(E)


def get_real_cardinality(list_of_M):
    return len({item for M in list_of_M for item in M})


def compute_err(real, estimated):
    return (abs(real - estimated) / real) * 100


def test():
    print("Generating datasets...")
    data = [[randint(0, 2**20) for i in range(10**7)] for j in range(5)]
    HLL_list = []
    estimated_error = 1.04 / sqrt(2**14) * 100
    print("Estimating cardinality...")
    start = time.time()
    for index, M in enumerate(data):
        print(f"Dataset {index + 1}")
        HLL = HyperLogLog(14, M)
        estimated = HLL.estimate_cardinality()
        real = len(set(M))
        print(f"Real cardinality: {real}")
        print(f"Estimated cardinality: {estimated}")
        print(f"Error: {compute_err(real, estimated)} %")
        print(f"Estimated error: {estimated_error} %")
        HLL.save(f"HLL{index + 1}")
        print(
            f"HLL{index + 1} size: {os.path.getsize(f'HLL{index + 1}.pickle') // 1024} kB"
        )
        HLL_list.append(HLL)
        print("--------------------------------------------------")
        print()

    print("Merging HLLs...")
    merged_S = merge_HLL(HLL_list)
    print("Estimating cardinality...")
    estimated = estimate_cardinality_merged(merged_S)
    real = get_real_cardinality(data)
    print(f"Real cardinality: {real}")
    print(f"Estimated cardinality: {estimated}")
    print(f"Error: {compute_err(real, estimated)} %")
    print(f"Estimated error: {estimated_error} %")
    end = time.time()
    print(f"Time: {end - start} s")


if __name__ == "__main__":
    test()
