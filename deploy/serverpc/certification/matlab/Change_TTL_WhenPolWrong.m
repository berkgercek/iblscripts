function [cell_ts_i_stim, nTTL_Stim_expected_i_stim] = Change_TTL_WhenPolWrong(ID_stim_i_stim, is_pol_start_ok, is_pol_end_ok, cell_ts_i_stim, nTTL_Stim_expected_i_stim)
% Change in case polarity is wrong

if is_pol_start_ok == 0
    switch ID_stim_i_stim
        case 1
            cell_ts_i_stim = cell_ts_i_stim(2:end);
            warning('--> 1 TTL pulse removed at begining')
        case 2
            cell_ts_i_stim = cell_ts_i_stim(2:end);
            warning('--> 1 TTL pulse removed at begining')
        case 3
            
        case 4
            cell_ts_i_stim = cell_ts_i_stim(2:end);
            warning('--> 1 TTL pulse removed at begining')
    end
end



if is_pol_end_ok == 0
    switch ID_stim_i_stim
        case 1
            cell_ts_i_stim = cell_ts_i_stim(1:end-1);
            warning('--> 1 TTL pulse removed at end')
        case 2
            nTTL_Stim_expected_i_stim = nTTL_Stim_expected_i_stim - 1;
            warning('--> n TTL expected -1')
        case 3
            
        case 4
            cell_ts_i_stim = cell_ts_i_stim(1:end-1);
            warning('--> 1 TTL pulse removed at end')
    end
end

end

