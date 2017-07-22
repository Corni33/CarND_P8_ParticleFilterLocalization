/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#define _USE_MATH_DEFINES

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 200;

  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_psi(theta, std[2]);

  // create particles using initial GPS estimate 
  for (int i = 0; i < num_particles; ++i)
  {
    Particle p;

    p.id = i;
    p.weight = 1.0;

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_psi(gen);

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  default_random_engine gen;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i=0; i < num_particles; ++i)
  {
    Particle *p = &particles[i];

    if (fabs(yaw_rate) < 0.0000001) 
    {           
      p->x += velocity * delta_t * cos(p->theta);
      p->y += velocity * delta_t * sin(p->theta);
      //p->theta += 0.0;
    }
    else
    {
      p->x += velocity / yaw_rate *(sin(p->theta + yaw_rate*delta_t) - sin(p->theta))+dist_x(gen);
      p->y += velocity / yaw_rate *(cos(p->theta) - cos(p->theta + yaw_rate*delta_t))+dist_y(gen);
      p->theta += yaw_rate*delta_t+dist_theta(gen);
    }      

  } 
}

/*void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.
}*/

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
  // update particle weights according to the likelihood, that the measured observations occur given the current state of the particle

  double dx, dy, var_x, var_y, error_x, error_y, likelihood, dist_squared;

  var_x = std_landmark[0] * std_landmark[0];
  var_y = std_landmark[1] * std_landmark[1];

  LandmarkObs map_obs, *obs;
  Map::single_landmark_s *closest_landmark;

  for (int i = 0; i < num_particles; ++i)
  {
    Particle *p = &particles[i];

    likelihood = 1.0;

    for (int k = 0; k < observations.size(); ++k)
    {
      obs = &observations[k]; // current landmark observation

      // transform observation into map coordinate system      
      map_obs.x = p->x + cos(p->theta) * obs->x - sin(p->theta) * obs->y;
      map_obs.y = p->y + sin(p->theta) * obs->x + cos(p->theta) * obs->y;

      // find closest landmark 
      double shortest_dist_squared = sensor_range*sensor_range;
      for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
      {        
        dx = map_landmarks.landmark_list[j].x_f - map_obs.x;
        dy = map_landmarks.landmark_list[j].y_f - map_obs.y;
        dist_squared = dx*dx + dy*dy;
        
        if (dist_squared < shortest_dist_squared)
        {
          // save closest landmark
          closest_landmark = &map_landmarks.landmark_list[j];
          shortest_dist_squared = dist_squared;
        }

      } // landmarks

      // calculate difference between expected and actual observation 
      error_x = map_obs.x - closest_landmark->x_f; 
      error_y = map_obs.y - closest_landmark->y_f;

      // calculate observation likelihood based on a 2D normal distribution
      likelihood *= 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]) *
        exp(-0.5*(error_x*error_x / var_x + error_y*error_y / var_y));

    } // measurements

    p->weight = likelihood;  

  } // particles

}

void ParticleFilter::resample() {

  default_random_engine gen;

  vector<Particle> current_particles(particles);

  std::vector<double> weights(num_particles);
  for (int i = 0; i < num_particles; ++i)
  {
    weights[i] = current_particles[i].weight;
  }

  //create a discrete distribution according to particle weights
  discrete_distribution<int> sample_particle(weights.begin(), weights.end());

  //sample particles
  for (int i = 0; i < num_particles; ++i)
  {
    particles[i] = current_particles[sample_particle(gen)];
  }


  // alternative resampling strategy 

  /*vector<Particle> current_particles(particles);

  double r = ((double) rand() / (RAND_MAX));
  int index = (int)(r * num_particles);
  double beta = 0.0, mw = 0.0;

  // calculate biggest weight mw
  for (int i = 0; i < num_particles; ++i)
  {
    if (current_particles[i].weight > mw) mw = current_particles[i].weight;
  }

  // sample according to the "sampling wheel" strategy
  for (int i = 0; i < num_particles; ++i)
  {
    r = ((double)rand() / (RAND_MAX));
    beta += r * 2.0 * mw;
    while (beta > current_particles[index].weight)
    {
      beta -= current_particles[index].weight;
      index = (index + 1) % num_particles;
    }

    particles[i] = current_particles[index];
  }*/  

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
