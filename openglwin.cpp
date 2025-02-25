#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>

// Constants
const int GRID_WIDTH = 960; // Number of columns
const int GRID_HEIGHT = 540; // Number of rows
const int CELL_SIZE = 2;
const int WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE;
const int WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE;

int *grid;      // Current grid state
int *new_grid;  // Temporary grid for next generation

const int thread_count = 8;

void setupProjection()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1); // Orthographic projection
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void drawGridBackground()
{
    glColor3f(0.34f, 0.34f, 0.34f); // Dark gray color for the grid background
    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glVertex2f(WINDOW_WIDTH, 0);
    glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT);
    glVertex2f(0, WINDOW_HEIGHT);
    glEnd();
}

void drawParticle(int x, int y, int type)
{
    if (type == 1)
    {
        glColor3f(1.0f, 1.0f, 0.0f); // Yellow
    }
    else
    {
        return;
    }

    int screenX = x * CELL_SIZE;
    int screenY = y * CELL_SIZE;

    glBegin(GL_QUADS);
    glVertex2f(screenX, screenY);
    glVertex2f(screenX + CELL_SIZE, screenY);
    glVertex2f(screenX + CELL_SIZE, screenY + CELL_SIZE);
    glVertex2f(screenX, screenY + CELL_SIZE);
    glEnd();
}

int count_neighbors(int x, int y, int *grid, int rows, int cols)
{
    int count = 0;
    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            if (dx != 0 || dy != 0) // Exclude the current cell
            {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < rows && ny >= 0 && ny < cols)
                {
                    if (grid[nx * cols + ny] == 1)
                        count++;
                }
            }
        }
    }
    return count;
}

void swap_ptr(int **p1, int **p2) // Used to swap grids between generations
{
    int *tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
}

void updateGrid()
{
    #pragma omp parallel num_threads(thread_count)
    {
        #pragma omp single // Ensure tasks are created by a single thread
        {
            for (int i = 0; i < GRID_HEIGHT; ++i)
            {
                #pragma omp task firstprivate(i)
                {
                    for (int j = 0; j < GRID_WIDTH; ++j)
                    {

                        new_grid[i * GRID_WIDTH + j] = 0; // Default to empty cell

                        int neighbors = count_neighbors(i, j, grid, GRID_HEIGHT, GRID_WIDTH);

                        if (grid[i * GRID_WIDTH + j] == 1)
                        { // If the current cell is alive
                            if (neighbors == 2 || neighbors == 3)
                                new_grid[i * GRID_WIDTH + j] = 1; // Survives
                        }
                        else
                        { // If the current cell is dead
                            if (neighbors == 3)
                                new_grid[i * GRID_WIDTH + j] = 1; // Becomes alive
                        }
                            
                    }
                }
            }
        }   // End of single section
    }   // End of parallel region
    
    swap_ptr(&grid, &new_grid); // Swap grids for the next generation
}

void renderGrid()
{
    glClear(GL_COLOR_BUFFER_BIT); // Clear the screen
    drawGridBackground(); // Draw the grid background

    // Draw particles
    for (int y = 0; y < GRID_HEIGHT; ++y)
    {
        for (int x = 0; x < GRID_WIDTH; ++x)
        {
            drawParticle(x, y, grid[y * GRID_WIDTH + x]);
        }
    }

    glfwSwapBuffers(glfwGetCurrentContext());
}

void initializeGridFromFile(const char *filename, int *grid, int rows, int cols)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    for (int i = 0; i < rows; ++i)
    {
        if (!std::getline(file, line))
        {
            std::cerr << "Error: Missing or invalid data in the file!" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < cols; ++j)
        {
            if (line[j] == '.')
            {
                grid[i * cols + j] = 0;
            }
            else if (line[j] == 'X')
            {
                grid[i * cols + j] = 1;
            }
            else
            {
                std::cerr << "Error: Invalid character '" << line[j] << "' in the file!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    file.close();
}

int main(void)
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Get the primary monitor for fullscreen
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    if (!monitor)
    {
        std::cerr << "Failed to get primary monitor" << std::endl;
        glfwTerminate();
        return -1;
    }

    const GLFWvidmode *mode = glfwGetVideoMode(monitor);

    // Set fullscreen mode
    GLFWwindow *window = glfwCreateWindow(mode->width, mode->height, "Cellular Automata Simulation", monitor, NULL);
    if (!window)
    {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window" << std::endl;
        return -1;
    }

    glfwMakeContextCurrent(window);
    glewInit();

    setupProjection();

    // Set the viewport to match the screen resolution
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    glfwSwapInterval(1); // Enable V-Sync

    // Allocate memory for the grids
    grid = new int[GRID_WIDTH * GRID_HEIGHT];
    new_grid = new int[GRID_WIDTH * GRID_HEIGHT];

    // Load the grid from a file
    initializeGridFromFile("bacteria1000.txt", grid, GRID_HEIGHT, GRID_WIDTH);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Check if Escape is pressed to close the program
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Update and render the grid
        updateGrid();
        renderGrid();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    delete[] grid;
    delete[] new_grid;

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
