import tlc

table = tlc.Table.from_url("/home/z/zhangyon/.local/share/3LC/projects/Visual_Entailment_3LC/datasets/VE_3lc_Model_Comparison_final/tables/SetGoldLabelIn103RowsTo3Values")
table.export("table_nach_labelkorrektion.csv")