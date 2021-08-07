#define _GNU_SOURCE
#include <unistd.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "intercept.h"

typedef ssize_t (*execve_func_t)  (const char* filename, char* const argv[], char* const envp[]);
//typedef ssize_t (*execl_func_t)   (const char* path, const char *arg, ...);
//typedef ssize_t (*execlp_func_t)  (const char* file, const char *arg, ...);
//typedef ssize_t (*execle_func_t)  (const char *path, const char *arg, ..., char * const envp[]);
typedef ssize_t (*execv_func_t)   (const char* path, char* const argv[]);
typedef ssize_t (*execvp_func_t)  (const char* file, char* const argv[]);
typedef ssize_t (*execvpe_func_t) (const char *file, char *const argv[], char *const envp[]);

static execve_func_t    old_execve = NULL;
//static execl_func_t     old_execl = NULL;
//static execlp_func_t    old_execlp = NULL;
//static execle_func_t    old_execle = NULL;
static execv_func_t     old_execv = NULL;
static execvp_func_t    old_execvp = NULL;
static execvpe_func_t   old_execvpe = NULL;

static const char *nvcc_fpc = NVCC_WRAPPER;
static const char *clang_fpc = CLANG_WRAPPER;
static const char *clangpp_fpc = CLANGPP_WRAPPER;
static const char *mpi_fpc = MPI_WRAPPER;
static const char *mpipp_fpc = MPIPP_WRAPPER;

/** Return: one if the string t occurs at the end of the string s, and zero otherwise **/
int str_end(const char *s, const char *t)
{
    if (strlen(s) < strlen(t)) return 0;
    return 0 == strcmp(&s[strlen(s)-strlen(t)], t);
}

int isNVCC(const char* filename) {
  return (str_end(filename, "/nvcc") ||
          strcmp(filename, "nvcc")==0
          );
}

int isClang(const char* filename) {
  return (str_end(filename, "/clang") || 
          strcmp(filename, "clang")==0
          );
}

int isClangPP(const char* filename) {
  return (str_end(filename, "/clang++") || 
          strcmp(filename, "clang++")==0
          );
}

int isMPI(const char* filename) {
  return (str_end(filename, "/mpicc") || 
          strcmp(filename, "mpicc")==0
          );
}

int isMPIPP(const char* filename) {
  return (str_end(filename, "/mpicxx") || 
          str_end(filename, "/mpic++") ||
          strcmp(filename, "mpicxx")==0 ||
          strcmp(filename, "mpic++")==0
          );
}

void printEnvironment(char* const envp[]) {
  size_t elems = 0;
  while (envp != NULL) {
    if (*envp == NULL)
      break;

    elems++;
    printf("VAR: %s\n", *envp);
    envp++;
  }
  printf("Elems: %lu\n", elems);
}

/** Copy the environment without LD_PRELOAD **/
void copy_env_variables(char* const envp[], char *** new_envp) {
  char **ptr = (char **)envp;
  size_t elems = 0;
  while (ptr != NULL) {
    if (*ptr == NULL)
      break;
    elems++;
    ptr++;
  }

  *new_envp = (char **)malloc(sizeof(char *)*elems+1); 
  for (size_t i=0; i < elems; ++i) {
    (*new_envp)[i] = (char *)malloc(strlen(envp[i]) * sizeof(char) + 1);
    if (strstr (envp[i], "LD_PRELOAD=") == NULL) { // do not copy ld_preload
      strcpy((*new_envp)[i], envp[i]);
    } else {
      strcpy((*new_envp)[i], "LD_PRELOAD=");
    }
  }
  (*new_envp)[elems] = NULL;
}

void remove_ld_preload() {
    //char *new_env = (char *)malloc(sizeof(char));
    //new_env[0] = '\n';
    //size_t elems = 0;
    char **ptr = environ;
    while (ptr != NULL) {
      if (*ptr == NULL)
        break;
      if (strstr(*ptr, "LD_PRELOAD=") != NULL) {
        //printf("....found LD_PRELOAD: %s\n", *ptr);
        strcpy(*ptr, "LD_PRELOAD=");
        break;
      }
      ptr++;
    }
}

int execve(const char* filename, char* const argv[], char* const envp[]) {
    //printf("In  execve::  %s\n", filename);
    // Copy env variables
    char ** new_envp;
    copy_env_variables(envp, &new_envp);
    //printEnvironment(envp);
    old_execve = dlsym(RTLD_NEXT, "execve");

    if (isNVCC(filename))         return old_execve(nvcc_fpc, argv, new_envp);
    else if (isClang(filename))   return old_execve(clang_fpc, argv, new_envp);
    else if (isClangPP(filename)) return old_execve(clangpp_fpc, argv, new_envp);
    else if (isMPI(filename))     return old_execve(mpi_fpc, argv, new_envp);
    else if (isMPIPP(filename))   return old_execve(mpipp_fpc, argv, new_envp);
    return old_execve(filename, argv, envp); // else run original call
}

/*int execl(const char *path, const char *arg, ...) {
    printf("In execl: %s\n", path);
    old_execl = dlsym(RTLD_NEXT, "execl");

    if (isNVCC(filename))       return old_execl(nvcc_fpc, arg);
    else if (isClang(filename)) return old_execl(clang_fpc, arg);
    else if (isMPI(filename))   return old_execl(mpi_fpc, argv);
    return old_execl(path, argv); // else run original call
}*/

/*int execlp(const char *file, const char *arg, ...) {
    printf("in execlp: %s\n", file);
    return 0;
}*/

/*int execle(const char *path, const char *arg, ..., char * const envp[]) {
    printf("In execle: %s\n", path);
    return 0;
}*/

int execv(const char *path, char *const argv[]) {
    //printf("In execv: %s\n", path);
    if (isNVCC(path) || isClang(path) || isClangPP(path) || isMPI(path) || isMPIPP(path))
      remove_ld_preload();
    old_execv = dlsym(RTLD_NEXT, "execv");

    if (isNVCC(path))         return old_execv(nvcc_fpc, argv);
    else if (isClang(path))   return old_execv(clang_fpc, argv);
    else if (isClangPP(path)) return old_execv(clangpp_fpc, argv);
    else if (isMPI(path))     return old_execv(mpi_fpc, argv);
    else if (isMPIPP(path))   return old_execv(mpipp_fpc, argv);
    return old_execv(path, argv); // else run original call
}

int execvp (const char *file, char *const argv[]) {
    //printf("In execvp: %s\n", file);
    if (isNVCC(file) || isClang(file) || isClangPP(file) || isMPI(file) || isMPIPP(file))
      remove_ld_preload();
    old_execvp = dlsym(RTLD_NEXT, "execvp");

    if (isNVCC(file))         return old_execvp(nvcc_fpc, argv);
    else if (isClang(file))   return old_execvp(clang_fpc, argv);
    else if (isClangPP(file)) return old_execvp(clangpp_fpc, argv);
    else if (isMPI(file))     return old_execvp(mpi_fpc, argv);
    else if (isMPIPP(file))   return old_execvp(mpipp_fpc, argv);
    return old_execvp(file, argv); // else run original call
}

int execvpe(const char *file, char *const argv[], char *const envp[]) {
    //printf("in execvpe: %s\n", file);
    char ** new_envp;
    copy_env_variables(envp, &new_envp);
    old_execvpe = dlsym(RTLD_NEXT, "execvpe");

    if (isNVCC(file))         return old_execvpe(nvcc_fpc, argv, new_envp);
    else if (isClang(file))   return old_execvpe(clang_fpc, argv, new_envp);
    else if (isClangPP(file)) return old_execvpe(clangpp_fpc, argv, new_envp);
    else if (isMPI(file))     return old_execvpe(mpi_fpc, argv, new_envp);
    else if (isMPIPP(file))   return old_execvpe(mpipp_fpc, argv, new_envp);
    return old_execvpe(file, argv, envp); // else run original call
}

//__attribute__((constructor)) static void setup(void) {
//  printf("In setup()\n");
//}


