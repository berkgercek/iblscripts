function [is_pol_start_ok,is_pol_end_ok] = TestStimPolarityIntegrity(pol_start_expected, pol_end_expected, vect_polarity_sort,id_ts_stim)
% Test for stimulus polarity integrity at begining and end
% pol_start_expected, pol_end are expected polarities
% Initialisation
is_pol_start_ok = boolean(1);
is_pol_end_ok = boolean(1);

% Test start
if vect_polarity_sort(id_ts_stim(1)) ~= pol_start_expected
    is_pol_start_ok = 0 ;
end

% Test end
if vect_polarity_sort(id_ts_stim(end)) ~= pol_end_expected
    is_pol_end_ok = 0 ;
end

end

