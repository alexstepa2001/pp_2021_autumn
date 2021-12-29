// Copyright 2021 Stepanov Alexander
#include <mpi.h>
#include <vector>
#include <random>
#include <stack>
#include <algorithm>
#include "../../../modules/task_3/stepanov_a_convex_hull_binary_image/convex_hull_binary_image.h"


void generateBinaryImage(std::vector<int>* image,
    std::size_t count_rows, std::size_t count_columns) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> urd(0, 1);

    for (std::size_t i = 0; i < count_columns * count_rows; i++) {
        auto random_value = urd(gen);
        (*image)[i] = random_value > 0.5 ? 1 : 0;
    }
}

std::vector<std::vector<int>> markingComponents(std::vector<int>* image,
    std::size_t count_rows, std::size_t count_columns) {
    std::size_t last_marker = 0;
    std::vector<std::vector<int>> container_components;

    for (std::size_t index = 0; index < count_columns * count_rows; index++) {
        if ((*image)[index] == 1) {
            auto components = markingNeighbors(image, count_rows, count_columns,
                static_cast<int>(last_marker + 2), index);

            container_components.push_back(components);
            last_marker++;
        }
    }

    return container_components;
}

std::vector<int> markingNeighbors(std::vector<int>* image,
    std::size_t count_rows, std::size_t count_columns,
    int marker, std::size_t start_index) {
    std::vector<int> points;
    std::stack<std::size_t> pixel_stack;
    pixel_stack.push(start_index);
    (*image)[start_index] = marker;

    while (pixel_stack.size() > 0) {
        std::size_t pos = pixel_stack.top();
        points.push_back(static_cast<int>(pos));
        pixel_stack.pop();

        std::size_t right_pos = pos + 1,
        down_pos = pos + count_columns,
        left_pos = pos - 1,
        upper_pos = pos - count_columns;

        if (right_pos < count_rows * count_columns && right_pos % count_columns > 0 &&
            (*image)[right_pos] == 1) {
            (*image)[right_pos] = marker;
            pixel_stack.push(right_pos);
        }

        if (down_pos < count_rows * count_columns && (*image)[down_pos] == 1) {
            (*image)[down_pos] = marker;
            pixel_stack.push(down_pos);
        }


        if (pos > 0 && pos % count_columns > 0 && (*image)[left_pos] == 1) {
            (*image)[left_pos] = marker;
            pixel_stack.push(left_pos);
        }

        if (pos >= count_columns && (*image)[upper_pos] == 1) {
            (*image)[upper_pos] = marker;
            pixel_stack.push(upper_pos);
        }
    }
    return points;
}

std::vector<std::vector<int>> createHullImageSequential(const std::vector<int>& image,
    std::size_t count_rows, std::size_t count_columns) {
    std::vector<int> marking_image = image;
    auto container_components = markingComponents(&marking_image, count_rows, count_columns);

    std::vector<std::vector<int>> container_hulls(container_components.size(), std::vector<int>{});

    for (std::size_t i = 0; i < container_components.size(); i++) {
        container_hulls[i] = createHullComponent(container_components[i], count_rows, count_columns);
    }

    return container_hulls;
}

int rotate(int a, int b, int c,
    std::size_t count_rows_utype, std::size_t count_columns_utype) {
    int count_rows = static_cast<int>(count_rows_utype);
    int count_columns = static_cast<int>(count_columns_utype);

    int a_x = a % count_columns;
    int a_y = count_rows - a / count_columns;

    int b_x = b % count_columns;
    int b_y = count_rows - b / count_columns;

    int c_x = c % count_columns;
    int c_y = count_rows - c / count_columns;

    return (b_x - a_x) * (c_y - b_y) - (b_y - a_y) * (c_x - b_x);
}

std::vector<int> createHullComponent(std::vector<int> points,
    std::size_t count_rows, std::size_t count_columns) {
    if (points.size() <= 2)
        return points;

    int p0 = points[0];
    for (int p : points) {
        if (p % count_columns < p0 % count_columns)
            p0 = p;

        if (p % count_columns == p0 % count_columns && p / count_columns > p0 / count_columns)
            p0 = p;
    }

    std::vector<int> hull = { p0 };
    int next_point = -1;
    int last_point = -1;
    auto deleted_iter = points.begin();

    while (next_point != p0) {
        last_point = *(--hull.end());
        next_point = points[0];
        deleted_iter = points.begin();

        for (std::size_t i = 1; i < points.size(); i++) {
            int rot = rotate(last_point, next_point, points[i], count_rows, count_columns);
            if (rot < 0) {
                next_point = points[i];
                deleted_iter = points.begin() + i;
            }
        }

        hull.push_back(next_point);
        points.erase(deleted_iter);
    }

    return hull;
}

std::vector<std::vector<int>> createHullImageParallel(const std::vector<int>& image,
    std::size_t count_rows, std::size_t count_columns) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<int>> components;
    int local_size = 0, remains = 0;
    if (rank == 0) {
        std::vector<int> marking_image = image;

        components = markingComponents(&marking_image, count_rows, count_columns);
        local_size = components.size() / size;
        remains = components.size() % size;
    }

    MPI_Bcast(&local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<std::vector<int>> local_components(local_size + remains);

    if (rank == 0) {
        for (int proc = 1; proc < size; proc++) {
            int step = local_size * proc + remains;
            for (int i = 0; i < local_size; i++) {
                int count_points = components[step + i].size();

                MPI_Send(&count_points, 1, MPI_INT, proc, i * 2, MPI_COMM_WORLD);
                MPI_Send(components[step + i].data(), count_points, MPI_INT, proc, i * 2 + 1, MPI_COMM_WORLD);
            }
        }

        for (int i = 0; i < local_size + remains; i++) {
            local_components[i] = components[i];
        }
    } else {
        for (int i = 0; i < local_size; i++) {
            int count_points;
            MPI_Recv(&count_points, 1, MPI_INT, 0, i * 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            local_components[i].resize(count_points);
            MPI_Recv(local_components[i].data(), count_points, MPI_INT, 0, i * 2 + 1,
                MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }
    }

    std::vector<std::vector<int>> local_hulls;
    int count_points = 0;
    for (std::size_t i = 0; i < static_cast<int>(local_size + remains); i++) {
            local_hulls.push_back(createHullComponent(local_components[i], count_rows, count_columns));
            count_points += (local_hulls[i].size() + 1);
    }

    int max_count_point = 0;
    MPI_Allreduce(&count_points, &max_count_point, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    std::vector<int> local_hulls_points(max_count_point, -1);
    std::size_t index = 0;
    for (std::size_t i = 0; i < static_cast<int>(local_size + remains); i++) {
        for (std::size_t j = 0; j < local_hulls[i].size(); j++) {
            local_hulls_points[index++] = local_hulls[i][j];
        }
        local_hulls_points[index++] = -1;
    }

    std::vector<int> global_hulls_points(max_count_point * size);
    MPI_Gather(local_hulls_points.data(), max_count_point, MPI_INT,
        global_hulls_points.data(), max_count_point, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::vector<int>> global_hulls;
    if (rank == 0) {
        std::vector<int> hull;
        for (std::size_t i = 0; i < global_hulls_points.size(); i++) {
            if (global_hulls_points[i] == -1 && hull.size() > 0) {
                global_hulls.push_back(hull);
                hull.clear();
            }

            if (global_hulls_points[i] != -1) {
                hull.push_back(global_hulls_points[i]);
            }
        }
    }

    return global_hulls;
}
