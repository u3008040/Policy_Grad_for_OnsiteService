import pandas as pd

# 定义一个函数，用于根据零售商索引读取对应的销售数据
def load_sales_data(retailer_index, file_path):
    # 创建零售商索引与工作表名称的映射
    retailer_sheet_mapping = {
        0: 'Re1',
        1: 'Re2',
        2: 'Re3'
    }

    # 获取对应的工作表名称
    sheet_name = retailer_sheet_mapping.get(retailer_index)
    if sheet_name is None:
        raise ValueError(f"无效的零售商索引：{retailer_index}")

    # 读取指定工作表的数据
    sales_data = pd.read_excel(file_path, sheet_name=sheet_name)

    # 将'日期'列转换为datetime类型
    sales_data['日期'] = pd.to_datetime(sales_data['日期'], format='%Y/%m/%d')

    # 确保'是否是节假日'列为整数类型
    sales_data['是否是节假日'] = sales_data['是否是节假日'].astype(int)

    return sales_data

# 定义状态更新函数
def state_update(current_date, current_period, sales_data):
    # 数据开始日期
    data_start_date = sales_data['日期'].min()

    # 计算距离数据开始日期的天数
    days_since_start = (current_date - data_start_date).days
    is_first_30_days = 1 if days_since_start < 30 else 0

    # 计算过去一个月的日期范围
    past_month_start = current_date - pd.Timedelta(days=30)
    past_month_end = current_date

    # 始终计算过去一个月的销量，无论数据是否足够
    past_month_data = sales_data[
        (sales_data['日期'] >= past_month_start) &
        (sales_data['日期'] < past_month_end)
    ]

    past_month_morning_sales = past_month_data['早销量'].sum()
    past_month_noon_sales = past_month_data['午销量'].sum()
    past_month_evening_sales = past_month_data['晚销量'].sum()

    # 今天的数据
    today_data = sales_data[sales_data['日期'] == current_date]

    # 获取今天的节假日信息
    is_holiday = today_data['是否是节假日'].values[0] if not today_data.empty else 0

    # 今天的销量（根据当前时间段）
    todays_morning_sales = 0
    todays_noon_sales = 0

    # 如果当前时间段是中午或晚上，可以获取今天早上的销量
    if current_period in ['noon', 'evening']:
        todays_morning_sales = today_data['早销量'].values[0] if not today_data.empty else 0

    # 如果当前时间段是晚上，可以获取今天中午的销量
    if current_period == 'evening':
        todays_noon_sales = today_data['午销量'].values[0] if not today_data.empty else 0

    # 昨天的日期
    yesterday = current_date - pd.Timedelta(days=1)

    # 昨天的销量
    yesterday_data = sales_data[sales_data['日期'] == yesterday]
    yesterday_morning_sales = yesterday_data['早销量'].values[0] if not yesterday_data.empty else 0
    yesterday_noon_sales = yesterday_data['午销量'].values[0] if not yesterday_data.empty else 0
    yesterday_evening_sales = yesterday_data['晚销量'].values[0] if not yesterday_data.empty else 0

    # 判断当前时间段
    is_morning = 1 if current_period == 'morning' else 0
    is_noon = 1 if current_period == 'noon' else 0
    is_evening = 1 if current_period == 'evening' else 0

    # 生成状态列表
    state = [
        past_month_morning_sales,  # 过去一个月早上的销量
        past_month_noon_sales,     # 过去一个月中午的销量
        past_month_evening_sales,  # 过去一个月晚上的销量
        is_first_30_days,          # 是否处于前30天
        is_holiday,                # 是否是节假日（周末、节日）
        todays_morning_sales,      # 今天早上的销量
        todays_noon_sales,         # 今天中午的销量
        yesterday_morning_sales,   # 昨天早上的销量
        yesterday_noon_sales,      # 昨天中午的销量
        yesterday_evening_sales,   # 昨天晚上的销量
        is_morning,                # 是否处于早上
        is_noon,                   # 是否处于中午
        is_evening                 # 是否处于晚上
    ]

    return state

# 示例调用
# 设置文件路径
file_path = r"D:\retailer_data\realdata.xlsx"

# 指定零售商索引
retailer_index = 1 # retailer 0 1 2

# 读取对应零售商的数据
sales_data = load_sales_data(retailer_index, file_path)

# 设置当前日期和时间段
current_date = pd.to_datetime('2019/12/15')
current_period = 'evening'  # 'morning', 'noon', or 'evening'

# 调用状态更新函数
state = state_update(current_date, current_period, sales_data)

# 输出状态列表
print(f"零售商 {retailer_index} 的当前日期：{current_date.date()}，当前时间段：{current_period}")
print(f"状态列表：{state}")
