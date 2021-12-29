// Copyright 2021 Stepanov Alexander
#ifndef MODULES_TASK_3_STEPANOV_A_CONVEX_HULL_BINARY_IMAGE_CONVEX_HULL_BINARY_IMAGE_H_
#define MODULES_TASK_3_STEPANOV_A_CONVEX_HULL_BINARY_IMAGE_CONVEX_HULL_BINARY_IMAGE_H_

#include <vector>

void generateBinaryImage(std::vector<int>* image,
    std::size_t count_rows, std::size_t count_columns);

std::vector<std::vector<int>> markingComponents(std::vector<int>* image,
    std::size_t count_rows, std::size_t count_columns);

std::vector<int> markingNeighbors(std::vector<int>* image,
    std::size_t count_rows, std::size_t count_columns,
    int marker, std::size_t start_index);

int rotate(int a, int b, int c,
    std::size_t count_rows, std::size_t count_columns);

std::vector<int> createHullComponent(std::vector<int> points,
    std::size_t count_rows, std::size_t count_columns);

std::vector<std::vector<int>> createHullImageSequential(const std::vector<int>& image,
    std::size_t count_rows, std::size_t count_columns);

std::vector<std::vector<int>> createHullImageParallel(const std::vector<int>& image,
    std::size_t count_rows, std::size_t count_columns);

#endif  // MODULES_TASK_3_STEPANOV_A_CONVEX_HULL_BINARY_IMAGE_CONVEX_HULL_BINARY_IMAGE_H_
