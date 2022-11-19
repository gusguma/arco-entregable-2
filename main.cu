///////////////////////////////////////////////////////////////////////////
/// PROGRAMACIÓN EN CUDA C/C++
/// Práctica:	ENTREGABLE 2 : Gráficos en CUDA
/// Autor:		Angel Sierra López, Gustavo Gutiérrez Martín
/// Fecha:		Noviembre 2022
///////////////////////////////////////////////////////////////////////////

/// Dependencias
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu_bitmap.h"
#include <ctime>

/// Constantes
#define MB (1<<20) /// MiB = 2^20
/// Dimension del bitmap horizontal
#define WIDTH 512
/// Dimension del bitmap vertical
#define HEIGHT 512
/// Numero de hilos
#define THREADS 16
/// Definimos el número de celdas en horizontal
#define CELLS_WIDTH 8
/// Definimos el número de celdas en vertical
#define CELLS_HEIGHT 8

/// Funciones
/// numero de CUDA cores
int getCudaCores(cudaDeviceProp deviceProperties);
/// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void kernel( unsigned char *imagen );

/// MAIN: rutina principal ejecutada en el host
int main() {
    /// almacena el número de devices disponibles
    int deviceCount;
    /// buscando dispositivos
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        /// mostramos el error si no se encuentra un dispositivo
        printf("¡No se ha encontrado un dispositivo CUDA!\n");
        printf("<pulsa [INTRO] para finalizar>");
        getchar();
        return 1;
    } else {
        ///obtenemos las propiedades del dispositivo CUDA
        int deviceId = 0;
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, deviceId);
        int SM = deviceProp.multiProcessorCount;
        int cudaCores = getCudaCores(deviceProp);
        printf("***************************************************\n");
        printf("DEVICE: %s\n", deviceProp.name);
        printf("***************************************************\n");
        printf("- Capacidad de Computo            \t: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("- No. de MultiProcesadores        \t: %d \n", SM);
        printf("- No. de CUDA Cores (%dx%d)       \t: %d \n", cudaCores, SM, cudaCores * SM);
        printf("- Memoria Global (total)          \t: %zu MiB\n", deviceProp.totalGlobalMem / MB);
        printf("- No. maximo de Hilos (por bloque)\t: %d\n", deviceProp.maxThreadsPerBlock);
        printf("***************************************************\n");
    }
    /// declaracion de eventos
    cudaEvent_t start;
    cudaEvent_t stop;
    /// Declaracion del bitmap:
    /// Inicializacion de la estructura RenderGPU
    RenderGPU foto(WIDTH, HEIGHT);
    /// Obtenemos el tamaño del bitmap en bytes
    size_t size = foto.image_size();
    /// Asignacion y reserva de la memoria en el host (framebuffer)
    unsigned char *host_bitmap = foto.get_ptr();
    /// Reserva de memoria en el device
    unsigned char *dev_bitmap;
    cudaMalloc( (void**)&dev_bitmap, size );
    /// Lanzamos un kernel bidimensional con bloques de 256 hilos (16x16)
    dim3 hilosB(THREADS,THREADS);
    /// Calculamos el numero de bloques necesario (un hilo por cada pixel)
    dim3 Nbloques(WIDTH/THREADS, HEIGHT/THREADS);
    /// creacion de eventos para calcular el tiempo de GPU
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /// marca inicio proceso GPU
    cudaEventRecord(start,0);
    /// Generamos el bitmap
    kernel<<<Nbloques,hilosB>>>( dev_bitmap );
    /// marca final proceso GPU
    cudaEventRecord(stop,0);
    /// sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    /// cálculo del tiempo de GPU en milisegundos
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    /// impresion de resultados
    printf("> Tiempo de ejecucion GPU: %f ms\n",elapsedTime);
    printf("***************************************************\n");
    /// Copiamos los datos desde la GPU hasta el framebuffer para visualizarlos
    cudaMemcpy( host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost );
    /// liberacion de recursos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /// función que muestra por pantalla la salida del programa
    time_t fecha;
    time(&fecha);
    printf("> Programa ejecutado el: %s", ctime(&fecha));
    printf("***************************************************\n");
    /// capturamos un INTRO para que no se cierre la consola de MSVS
    /// Visualizacion y salida
    printf("\n...pulsa [ESC] para finalizar...");
    foto.display_and_exit();
    return 0;
}

int getCudaCores(cudaDeviceProp deviceProperties) {
    int cudaCores = 0;
    int major = deviceProperties.major;
    if (major == 1) {
        /// TESLA
        cudaCores = 8;
    } else if (major == 2) {
        /// FERMI
        if (deviceProperties.minor == 0) {
            cudaCores = 32;
        } else {
            cudaCores = 48;
        }
    } else if (major == 3) {
        /// KEPLER
        cudaCores = 192;
    } else if (major == 5) {
        /// MAXWELL
        cudaCores = 128;
    } else if (major == 6 || major == 7 || major == 8) {
        /// PASCAL, VOLTA (7.0), TURING (7.5), AMPERE
        cudaCores = 64;
    } else {
        /// ARQUITECTURA DESCONOCIDA
        cudaCores = 0;
        printf("¡Dispositivo desconocido!\n");
    }
    return cudaCores;
}

__global__ void kernel( unsigned char *imagen ) {
    /// Kernel bidimensional multibloque
    /// coordenada horizontal de cada hilo
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    /// coordenada vertical de cada hilo
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    /// indice global de cada hilo (indice lineal para acceder a la memoria)
    unsigned int myID = x + y * blockDim.x * gridDim.x;
    /// cada hilo obtiene la posicion de su pixel
    unsigned int miPixel = myID * 4;
    /// calculamos en que posicion x dentro del tablero de 8x8 esta el pixel
    unsigned int positionX = (x * CELLS_WIDTH) / WIDTH;
    /// calculamos en que posicion y dentro del tablero de 8x8 esta el pixel
    unsigned int positionY = (y * CELLS_HEIGHT) / HEIGHT;
    /// cada hilo rellena los 4 canales de su pixel, si la positionX + la positionY es par, rellena de blanco, sino negro
    if ((positionX + positionY) % 2) {
        /// rellena el pixel de color blanco
        imagen[miPixel] = 255; /// canal R
        imagen[miPixel + 1] = 255;/// canal G
        imagen[miPixel + 2] = 255; /// canal B
        imagen[miPixel + 3] = 0; /// canal alfa
    } else {
        /// rellena el pixel de color negro
        imagen[miPixel] = 0; /// canal R
        imagen[miPixel + 1] = 0;/// canal G
        imagen[miPixel + 2] = 0; /// canal B
        imagen[miPixel + 3] = 0; /// canal alfa
    }
}
