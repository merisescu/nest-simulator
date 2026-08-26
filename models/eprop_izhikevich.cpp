/*
 *  eprop_izhikevich.cpp
 *
 *  Extension module by Pat Merisescu, 2026.
 * 
 *  This file is part of a user defined extension module for NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "eprop_izhikevich.h"

// C++
#include <limits>

// libnestutil
#include "dict_util.h"
#include "numerics.h"

// nestkernel
#include "eprop_archiving_node_recurrent_impl.h"
#include "exceptions.h"
#include "kernel_manager.h"
#include "nest_impl.h"
#include "universal_data_logger_impl.h"


namespace nest
{

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

RecordablesMap< eprop_izhikevich > eprop_izhikevich::recordablesMap_;

void register_eprop_izhikevich( const std::string& name )
{
    register_node_model< eprop_izhikevich >( name );
}

template <>
void RecordablesMap< eprop_izhikevich >::create()
{
    insert_( names::eprop_history_duration, &eprop_izhikevich::get_eprop_history_duration );
    insert_( names::V_m, &eprop_izhikevich::get_v_m_ );
    insert_( names::U_m, &eprop_izhikevich::get_u_m_ );
    insert_( names::learning_signal, &eprop_izhikevich::get_learning_signal_ );
    insert_( names::surrogate_gradient, &eprop_izhikevich::get_surrogate_gradient_ );
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

eprop_izhikevich::Parameters_::Parameters_()
    : a_( 0.02 )                                       // a
    , b_( 0.2 )                                        // b
    , c_( -65.0 )                                      // c without unit
    , d_( 8.0 )                                        // d
    , I_e_( 0.0 )                                      // pA
    , V_th_( 30.0 )                                    // mV
    , V_min_( -std::numeric_limits< double >::max() )  // mV
    , c_reg_( 0.0 )
    , f_target_( 0.01 )
    , surrogate_gradient_function_( "exponential" )
    , surrogate_gradient_height_( 0.3 )
    , surrogate_gradient_width_( 1.0 )
    , kappa_( 0.97 )
    , kappa_reg_( 0.97 )
{
}
// set const Parameters_& p as argument of state to include parameters in definition of initial state (prob useful)
eprop_izhikevich::State_::State_()
    : v_m_( -70.0 )        // membrane potential
    , u_m_( 0.2 * -70.0 )  // membrane recovery variable (b * V_m_init)
    , i_in_( 0.0 )          // input current
    , z_in_( 0 )          // n. input spikes (for eligibility trace calculation)
    , learning_signal_( 0.0 )
    , surrogate_gradient_( 0.0 )
{
}

eprop_izhikevich::Buffers_::Buffers_( eprop_izhikevich& n )
  : logger_( n )
{
}

eprop_izhikevich::Buffers_::Buffers_( const Buffers_&, eprop_izhikevich& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Getter and setter functions for parameters and state
 * ---------------------------------------------------------------- */

void
eprop_izhikevich::Parameters_::get( Dictionary& d ) const
{
  d[ names::I_e ] = I_e_;
  d[ names::V_th ] = V_th_;
  d[ names::V_min ] = V_min_;
  d[ names::a ] = a_;
  d[ names::b ] = b_;
  d[ names::c ] = c_;
  d[ names::d ] = d_;
  d[ names::c_reg ] = c_reg_;
  d[ names::f_target ] = f_target_;
  d[ names::surrogate_gradient_function ] = surrogate_gradient_function_;
  d[ names::surrogate_gradient_height ] = surrogate_gradient_height_;
  d[ names::surrogate_gradient_width ] = surrogate_gradient_width_;
  d[ names::kappa ] = kappa_;
  d[ names::kappa_reg ] = kappa_reg_;
}

void
eprop_izhikevich::Parameters_::set( const Dictionary& d, Node* node )
{
  update_value_param( d, names::I_e, I_e_, node );
  update_value_param( d, names::V_th, V_th_, node );
  update_value_param( d, names::V_min, V_min_, node );
  update_value_param( d, names::a, a_, node );
  update_value_param( d, names::b, b_, node );
  update_value_param( d, names::c, c_, node );
  update_value_param( d, names::d, d_, node );
  update_value_param( d, names::c_reg, c_reg_, node );

  if ( update_value_param( d, names::f_target, f_target_, node ) )
  {
    f_target_ /= 1000.0;  // convert from spikes/s to spikes/ms
  }

  if ( update_value_param( d, names::surrogate_gradient_function, surrogate_gradient_function_, node ) )
  {
    eprop_izhikevich* nrn = dynamic_cast< eprop_izhikevich* >( node );
    assert( nrn );
    nrn->compute_surrogate_gradient_ = nrn->find_surrogate_gradient( surrogate_gradient_function_ );
  }
  update_value_param( d, names::surrogate_gradient_width, surrogate_gradient_width_, node );
  update_value_param( d, names::surrogate_gradient_height, surrogate_gradient_height_, node );
  update_value_param( d, names::kappa, kappa_, node );
  update_value_param( d, names::kappa_reg, kappa_reg_, node );

  if ( V_th_ < V_min_ )
  {
    throw BadProperty( "V_th ≥ V_min required." );
  }

  if ( c_reg_ < 0 )
  {
    throw BadProperty( "c_reg ≥ 0 required." );
  }

  if ( f_target_ < 0 )
  {
    throw BadProperty( "f_target ≥ 0 required." );
  }

  if ( kappa_ < 0.0 or kappa_ > 1.0 )
  {
    throw BadProperty( "0 ≤ kappa ≤ 1 required." );
  }

  if ( kappa_reg_ < 0.0 or kappa_reg_ > 1.0 )
  {
    throw BadProperty( "0 ≤ kappa_reg ≤ 1 required." );
  }

  if ( surrogate_gradient_height_ <= 0.0 )
  {
    throw BadProperty( "surrogate_gradient_height > 0 required." );
  }

  if ( surrogate_gradient_width_ <= 0.0 )
  {
    throw BadProperty( "surrogate_gradient_width > 0 required." );
  }
}

void
eprop_izhikevich::State_::get( Dictionary& d, const Parameters_& p ) const
{
  d[ names::V_m ] = v_m_;
  d[ names::U_m ] = u_m_;
  d[ names::surrogate_gradient ] = surrogate_gradient_;
  d[ names::learning_signal ] = learning_signal_;
}

void
eprop_izhikevich::State_::set( const Dictionary& d, const Parameters_& p, Node* node )
{
  update_value_param( d, names::U_m, u_m_, node );
  update_value_param( d, names::V_m, v_m_, node );
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

eprop_izhikevich::eprop_izhikevich()
  : EpropArchivingNodeRecurrent()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
}

eprop_izhikevich::eprop_izhikevich( const eprop_izhikevich& n )
  : EpropArchivingNodeRecurrent( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

void
eprop_izhikevich::init_buffers_()
{
  B_.spikes_.clear();    // includes resize
  B_.currents_.clear();  // includes resize
  B_.logger_.reset();    // includes resize
}

void
eprop_izhikevich::pre_run_hook()
{
  B_.logger_.init();  // ensures initialization in case multimeter connected after Simulate

  FlushEventMechanism::pre_run_hook();

  const double dt = Time::get_resolution().get_ms();

  // calculate the entries of the propagator matrix for the evolution of the state vector

  V_.P_epsilon_v_ = dt * P_.a_ * P_.b_;
  V_.P_epsilon_ = 1 - dt * P_.a_;
  //V_.P_epsilon_v_t_ = dt * 0.08;
  //V_.P_epsilon_5_ = dt * 5;

}

/* ----------------------------------------------------------------
 * Update function
 * ---------------------------------------------------------------- */

void
eprop_izhikevich::update( Time const& origin, const long from, const long to )
{
  const double dt = Time::get_resolution().get_ms();
  double v_old, u_old;
  for ( long lag = from; lag < to; ++lag )
  {
    const long t = origin.get_steps() + lag;
    S_.z_in_ = B_.spike_count_.get_value( lag );
    S_.i_in_ = B_.spikes_.get_value( lag );
    v_old = S_.v_m_;
    u_old = S_.u_m_;


    S_.v_m_ += dt * ( 0.04*v_old*v_old + 5*v_old + 140 - u_old + S_.i_in_);
    S_.v_m_ = std::max( S_.v_m_, P_.V_min_ );

    S_.u_m_ += dt * ( P_.a_ * ( P_.b_ * v_old - u_old ) );

    double z = 0.0;  // spike state variable

    S_.surrogate_gradient_ = ( this->*compute_surrogate_gradient_ )(0, S_.v_m_, P_.V_th_, P_.surrogate_gradient_height_, P_.surrogate_gradient_width_ );

    if ( spike_event_is_due( S_.v_m_ >= P_.V_th_ ) )
    {
      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag );

      z = 1.0;
      S_.v_m_ = P_.c_;
      S_.u_m_ += P_.d_;
      set_last_event_time( t );
    }
    else if ( flush_event_is_due( t ) )
    {
      SpikeEvent se;
      se.set_flush_event_flag( true );
      kernel().event_delivery_manager.send( *this, se, lag );
      set_last_event_time( t );
    }

    append_new_eprop_history_entry( t );
    write_surrogate_gradient_to_history( t, S_.surrogate_gradient_ );
    write_firing_rate_reg_to_history( t, z, P_.f_target_, P_.kappa_reg_, P_.c_reg_ );
    write_voltage_to_history( t, S_.v_m_);
    write_spike_count_to_history( t, S_.z_in_ );

    S_.learning_signal_ = get_learning_signal_from_history( t );

    S_.i_in_ = B_.currents_.get_value( lag ) + P_.I_e_;

    B_.logger_.record_data( t );
  }
}

/* ----------------------------------------------------------------
 * Event handling functions
 * ---------------------------------------------------------------- */

void
eprop_izhikevich::handle( SpikeEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  B_.spike_count_.add_value( e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_multiplicity());

  B_.spikes_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() * e.get_multiplicity() );
}

void
eprop_izhikevich::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );

  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() * e.get_current() );
}

void
eprop_izhikevich::handle( LearningSignalConnectionEvent& e )
{
  for ( auto it_event = e.begin(); it_event != e.end(); )
  {
    const long time_step = e.get_stamp().get_steps();
    const double weight = e.get_weight();
    const double error_signal = e.get_coeffvalue( it_event );  // get_coeffvalue advances iterator
    const double learning_signal = weight * error_signal;

    write_learning_signal_to_history( time_step, learning_signal );
  }
}

void
eprop_izhikevich::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

void
eprop_izhikevich::compute_gradient( const long t_spike,
  const long t_spike_previous,
  double& z_previous_buffer,
  double& /*z_bar*/,
  double& e_bar,
  double& e_bar_reg,
  double& epsilon,
  double& epsilon_v,
  double& weight,
  const CommonSynapseProperties& cp,
  WeightOptimizer* optimizer,
  const bool is_flush_event,
  const bool previous_was_flush_event,
  double& gradient,
  long& remaining_steps_until_cutoff,
  long& decay_steps )
{
  const double dt = Time::get_resolution().get_ms();
  const auto& ecp = static_cast< const EpropSynapseCommonProperties& >( cp );
  const auto& opt_cp = *ecp.optimizer_cp_;
  const bool optimize_each_step = opt_cp.optimize_each_step_;

  const long isi_steps = t_spike - t_spike_previous;
  remaining_steps_until_cutoff = previous_was_flush_event ? remaining_steps_until_cutoff : get_eprop_isi_trace_cutoff();

  double z_current_buffer = 0.0;  // spike that triggered current computation
  if ( not previous_was_flush_event )
  {
    gradient = 0.0;  // gradient used for the weight update (to be calculated)
    z_current_buffer = 1.0;
  }

  const long t_begin = t_spike_previous - 1;
  auto eprop_hist_it = get_eprop_history( t_begin );
  const long t_steps = std::min( remaining_steps_until_cutoff, isi_steps );
  const long t_end = t_begin + t_steps;

  for ( long t = t_begin; t < t_end; ++t, ++eprop_hist_it )
  {
    require_eprop_history_entry( eprop_hist_it, t );

    const double z = z_previous_buffer;  // spiking variable
    z_previous_buffer = z_current_buffer;
    z_current_buffer = 0.0;

    const double psi = eprop_hist_it->surrogate_gradient_;  // surrogate gradient
    const double L = eprop_hist_it->learning_signal_;       // learning signal
    const double fr_reg = eprop_hist_it->firing_rate_reg_;  // firing rate regularization
    const double v_m = eprop_hist_it->v_m_; //membrane voltage
    const double z_in = eprop_hist_it->z_in_; // number of input spikes received per time step

    const double epsilon_v_old = epsilon_v; // temporary assignment to keep correct relations
    //const double epsilon_old = epsilon;
    const double z_nt = 1 - z;

    // TODO: maybe figure out how to remove z from this to cut down on the computation waste
    // bc for all except i think the second step (if previous was spike not flush event), then z = 0
    // i dont know what happens on t_end actually
    // maybe if condition it could be cheaper when compiled using t == t_spike_previous
    /*if ( t == t_spike_previous ) {
      epsilon_v = dt * ( S_.i_in_ - epsilon )
      epsilon = V_.P_epsilon_ * epsilon
    }
    else {
      epsilon_v = (1 + dt * ( 0.08 * v_m + 5 ) ) * epsilon_v - dt * ( epsilon + S_.i_in_);
      epsilon_ = ( V_.P_epsilon_v_ * epsilon_v_old + V_.P_epsilon_ * epsilon);
    }
    */
    epsilon_v = z_nt * (1 + dt * ( 0.08 * v_m + 5 ) ) * epsilon_v - dt * ( epsilon + z_in);
    epsilon = z_nt * ( V_.P_epsilon_v_ * epsilon_v_old + V_.P_epsilon_ * epsilon);

    const double e = psi * epsilon_v;  // eligibility trace

    e_bar = P_.kappa_ * e_bar + e;
    e_bar_reg = P_.kappa_reg_ * e_bar_reg + ( 1.0 - P_.kappa_reg_ ) * e;

    const double gradient_increment = L * e_bar + fr_reg * e_bar_reg;

    if ( optimize_each_step )
    {
      gradient = gradient_increment;
      weight = optimizer->optimized_weight( opt_cp, t + 1, gradient, weight );
    }
    else
    {
      gradient += gradient_increment;
    }
  }

  remaining_steps_until_cutoff -= t_steps;
  const long remaining_steps_until_event = isi_steps - t_steps;

  decay_steps += remaining_steps_until_event;

  if ( not is_flush_event and decay_steps > 0 )
  {
    e_bar *= std::pow( P_.kappa_, decay_steps );
    e_bar_reg *= std::pow( P_.kappa_reg_, decay_steps );
    decay_steps = 0;
  }

  if ( not is_flush_event and not optimize_each_step )
  {
    weight = optimizer->optimized_weight( opt_cp, t_end + remaining_steps_until_event, gradient, weight );
  }
}
} // namespace nest