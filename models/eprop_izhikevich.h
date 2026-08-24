/*
 *  eprop_izhikevich.h
 *
 *  Extension module by Pat Merisescu, 2026.
 * 
 *  This file is part of a user defined extension module for NEST.
 * 
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

#ifndef EPROP_IZHIKEVICH_H
#define EPROP_IZHIKEVICH_H

// nestkernel
#include "connection.h"
#include "eprop_archiving_node_impl.h"
#include "eprop_archiving_node_recurrent.h"
#include "eprop_synapse.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

namespace nest
{
    void register_eprop_izhikevich( const std::string& name );

/** @BeginDocumentation
    Name: eprop_izhikevich - Izhikevich neuron model with eligibility propagation mechanism

    Description:

    Transmits:

    Remarks:

    SeeAlso: izhikevich
*/

class eprop_izhikevich : public EpropArchivingNodeRecurrent< false >
{
public:
    //! constructor
    eprop_izhikevich();

    //! Copy constructor.
    eprop_izhikevich( const eprop_izhikevich& );

    using Node::handle;
    using Node::handles_test_event;

    size_t send_test_event( Node&, size_t, synindex, bool ) override;

    void handle( SpikeEvent& ) override;
    void handle( CurrentEvent& ) override;
    void handle( LearningSignalConnectionEvent& ) override;
    void handle( DataLoggingRequest& ) override;

    size_t handles_test_event( SpikeEvent&, size_t ) override;
    size_t handles_test_event( CurrentEvent&, size_t ) override;
    size_t handles_test_event( LearningSignalConnectionEvent&, size_t ) override;
    size_t handles_test_event( DataLoggingRequest&, size_t ) override;

    void get_status( Dictionary& ) const override;
    void set_status( const Dictionary& ) override;

private:

    void init_buffers_() override;
    void pre_run_hook() override;

    void update( Time const&, const long, const long ) override;

    void compute_gradient( const long,
    const long,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&,
    const CommonSynapseProperties&,
    WeightOptimizer*,
    const bool,
    const bool,
    double&,
    long&,
    long& ) override;

    long get_shift() const override;
    bool is_eprop_recurrent_node() const override;

    friend class RecordablesMap< eprop_izhikevich >;
    friend class UniversalDataLogger< eprop_izhikevich >;

    struct Parameters_
    {
        double a_;
        double b_;
        double c_;
        double d_;

        /** External DC current */
        double I_e_;

        /** Threshold */
        double V_th_;

        /** Lower bound */
        double V_min_;

        //! Coefficient of firing rate regularization.
        double c_reg_;

        //! Target firing rate of rate regularization (spikes/s).
        double f_target_;

        //! Surrogate gradient / pseudo-derivative function of the membrane voltage ["piecewise_linear", "exponential",
        //! "fast_sigmoid_derivative", "arctan_derivative"]
        std::string surrogate_gradient_function_;

        //! Height scaling of surrogate gradient / pseudo-derivative of membrane voltage.
        double surrogate_gradient_height_;

        //! Width scaling of surrogate gradient / pseudo-derivative of membrane voltage.
        double surrogate_gradient_width_;

        //! Low-pass filter of the eligibility trace.
        double kappa_;

        //! Low-pass filter of the firing rate for regularization.
        double kappa_reg_;

        //! Initialize parameters to their default values.
        Parameters_();

        //! Store parameter values in Dictionary.
        void get( Dictionary& ) const;

        //! Set parameter values from Dictionary.
        void set( const Dictionary&, Node* );
    };

    struct State_
    {
        double v_m_;
        double u_m_;
        double i_in_;

        double epsilon_v_;
        double epsilon_u_;

        double learning_signal_;
        double surrogate_gradient_;

        //! Default constructor.
        State_();

        //! Get the state variables and their values.
        void get( Dictionary&, const Parameters_& ) const;

        //! Set the state variables.
        void set( const Dictionary&, const Parameters_&, Node* );
    };

    struct Buffers_
    {
      //! Default constructor.
      Buffers_( eprop_izhikevich& );

      //! Copy constructor.
      Buffers_( const Buffers_&, eprop_izhikevich& );

      RingBuffer spikes_;
      RingBuffer currents_;
      UniversalDataLogger< eprop_izhikevich > logger_;

      /** buffers and sums up incoming spikes/currents */
      RingBuffer spikes_;
      RingBuffer currents_;
    };

    struct Variables_
    {
        //! Propagator matrix entry for evolving the membrane voltage (mathematical symbol "alpha" in user documentation).
        //double P_v_m_;

        //! Propagator matrix entry for evolving the adaptive current
        //double P_u_m_;

        //! Propagator matrix entry for evolving the incoming currents.
        //double P_i_in_;
    };
    //! Get the current value of the membrane voltage.
    double get_v_m_() const
    {
        return S_.v_m_;
    }

    double get_u_m_() const
    {
        return S_.u_m_;
    }

    //! Get the current value of the surrogate gradient.
    double get_surrogate_gradient_() const
    {
        return S_.surrogate_gradient_;
    }

    //! Get the current value of the learning signal.
    double get_learning_signal_() const
    {
        return S_.learning_signal_;
    }

    // the order in which the structure instances are defined is important for speed

    Parameters_ P_;
    State_ S_;
    Variables_ V_;
    Buffers_ B_;

    //! Map storing a static set of recordables.
    static RecordablesMap< eprop_izhikevich > recordablesMap_;

};


inline long
eprop_izhikevich::get_shift() const
{
  return offset_gen_ + delay_in_rec_;
}

inline bool
eprop_izhikevich::is_eprop_recurrent_node() const
{
  return true;
}

inline size_t
eprop_izhikevich::send_test_event( Node& target, size_t receptor_type, synindex, bool )
{
  SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline size_t
eprop_izhikevich::handles_test_event( SpikeEvent&, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw UnknownReceptorType( receptor_type, get_name() );
  }

  return 0;
}

inline size_t
eprop_izhikevich::handles_test_event( CurrentEvent&, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw UnknownReceptorType( receptor_type, get_name() );
  }

  return 0;
}

inline size_t
eprop_izhikevich::handles_test_event( LearningSignalConnectionEvent&, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw UnknownReceptorType( receptor_type, get_name() );
  }

  return 0;
}

inline size_t
eprop_izhikevich::handles_test_event( DataLoggingRequest& dlr, size_t receptor_type )
{
  if ( receptor_type != 0 )
  {
    throw UnknownReceptorType( receptor_type, get_name() );
  }

  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
eprop_izhikevich::get_status( Dictionary& d ) const
{
  EpropArchivingNodeRecurrent::get_status( d );
  P_.get( d );
  S_.get( d, P_ );
  d[ names::recordables ] = recordablesMap_.get_list();
}

inline void
eprop_izhikevich::set_status( const Dictionary& d )
{
    EpropArchivingNodeRecurrent::set_status( d );
    // temporary copies in case of errors
    Parameters_ ptmp = P_;
    ptmp.set( d, this ); 
    State_ stmp = S_;
    //stmp.set( d, ptmp, this );


    stmp.set( d, ptmp, this );

    P_ = ptmp;
    S_ = stmp;
}

} // namespace nest
#endif // EPROP_IZHIKEVICH_H