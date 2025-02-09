`ifndef N
`define N              16
`endif
`define W               8
`define lgN     ($clog2(`N))
`define dbLgN (2*$clog2(`N))

typedef struct packed { logic [`W-1:0] data; } data_t;

module add_(    // need 1 cycle to finish
    input   logic   clock,
    input   data_t  a,
    input   data_t  b,
    output  data_t  out
);
    always_ff @(posedge clock) begin
        out.data <= a.data + b.data;
    end
endmodule

module mul_(    // need 1 cycle to compute
    input   logic   clock,
    input   data_t  a,
    input   data_t  b,
    output  data_t out
);
    always_ff @(posedge clock) begin
        out.data <= a.data * b.data;
    end
endmodule

module RedUnit(
    input   logic               clock,
                                reset,
    input   data_t              data[`N-1:0],   // the input data here refers to the elements that have been multiplied but not correctly be added as the final result
    input   logic               split[`N-1:0],  // it present that the current calculation is over
    input   logic [`lgN-1:0]    out_idx[`N-1:0],
    output  data_t              out_data[`N-1:0],
    output  int                 delay,
    output  int                 num_el
);

    data_t prefix_sum[`N-1:0];
    data_t output_local[`N-1:0];
    data_t halo_adder;
    logic halo_delay;

    always_ff @(posedge clock or posedge reset) begin
        if(reset) begin
            for(int i = 0; i < `N; i++) begin
                prefix_sum[i] <= 0;
                output_local[i] <= 0;
                halo_delay <= 1;
            end
        end
        else begin
            for(int i = 0; i < `N; i++) begin
                if(i > 0) begin
                   if (split[i-1] == 1) begin
                        prefix_sum[i] <= data[i];
                   end
                   else begin
                        prefix_sum[i] <= prefix_sum[i-1] + data[i];
                   end
                end
                else begin
                    prefix_sum[i] <= data[i] + halo_adder;
                    halo_delay <= 0;
                end
                if (split[i] == 1) begin
                    output_local[out_idx[i]] <= prefix_sum[i];
                end
            end
            if (halo_delay == 0) begin
                if (split[`N-1] == 1) begin
                    halo_adder <= prefix_sum[`N-1];
                end
            end
            else begin
                halo_adder <= 0;
            end
        end
    end

    generate
        for(genvar i = 0; i < `N; i++) begin
            assign out_data[i] = output_local[i];
        end
    endgenerate

    // num_el 总是赋值为 N
    assign num_el = `N;
    // delay 你需要自己为其赋值，表示电路的延迟
    assign delay = `N;  // this value can be optimized later, now we set it 16 so as to guarantee the correctness of the result
endmodule

module PE(
    input   logic               clock,
                                reset,
    input   logic               lhs_start,
    input   logic [`dbLgN-1:0]  lhs_ptr [`N-1:0],
    input   logic [`lgN-1:0]    lhs_col [`N-1:0],
    input   data_t              lhs_data[`N-1:0],
    input   data_t              rhs[`N-1:0],
    output  data_t              out[`N-1:0],
    output  int                 delay,
    output  int                 num_el
);
    // num_el 总是赋值为 N
    assign num_el = `N;
    // delay 你需要自己为其赋值，表示电路的延迟
    assign delay = 0;

    generate
        for(genvar i = 0; i < `N; i++) begin
            assign out[i] = 0;
        end
    endgenerate
endmodule

module SpMM(
    input   logic               clock,
                                reset,
    /* 输入在各种情况下是否 ready */
    output  logic               lhs_ready_ns,
                                lhs_ready_ws,
                                lhs_ready_os,
                                lhs_ready_wos,
    input   logic               lhs_start,
    /* 如果是 weight-stationary, 这次使用的 rhs 将保留到下一次 */
                                lhs_ws,
    /* 如果是 output-stationary, 将这次的结果加到上次的 output 里 */
                                lhs_os,
    input   logic [`dbLgN-1:0]  lhs_ptr [`N-1:0],
    input   logic [`lgN-1:0]    lhs_col [`N-1:0],
    input   data_t              lhs_data[`N-1:0],
    output  logic               rhs_ready,
    input   logic               rhs_start,
    input   data_t              rhs_data [3:0][`N-1:0],
    output  logic               out_ready,
    input   logic               out_start,
    output  data_t              out_data [3:0][`N-1:0],
    output  int                 num_el
);
    // num_el 总是赋值为 N
    assign num_el = `N;

    assign lhs_ready_ns = 0;
    assign lhs_ready_ws = 0;
    assign lhs_ready_os = 0;
    assign lhs_ready_wos = 0;
    assign rhs_ready = 0;
    assign out_ready = 0;
endmodule
