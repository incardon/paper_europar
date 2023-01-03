#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
 
int main(){
  int sz = 32;
  int team_sz = 4;
  int nteams_required=sz*sz/team_sz, tm_id, th_id;

  double start; 
  double end; 
  start = omp_get_wtime(); 

  #pragma omp teams num_teams(nteams_required) thread_limit(team_sz) private(tm_id, th_id)
  {
    #pragma omp parallel num_threads(team_sz)
    // for (int team_rank = 0; team_rank < team_sz; ++team_rank) {
    {
      // tm_id = omp_get_team_num();
      // th_id = omp_get_thread_num();

      // printf("Thread id %d %d %d is running\n", omp_get_num_teams(), omp_get_team_num(), omp_get_thread_num());
      // printf("Thread id %d is running\n", omp_get_num_threads()*omp_get_team_num()+omp_get_thread_num());
      // if( omp_get_num_teams() != 2
      int k = omp_get_num_threads()*omp_get_team_num()+omp_get_thread_num();

      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier

      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier

      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier

      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
      #pragma omp barrier
    }
  }

  end = omp_get_wtime(); 

  printf("TIME: %f ns\n", (end - start) / (sz*sz*team_sz*24) * 1e9);
  return 0;
} 
