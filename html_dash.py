import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 결과 시각화 대시보드 생성 (HTML 파일로 저장)
def create_dashboard(class_ious_list, num_classes, avg_class_ious, filename='segmentation_dashboard.html'):
    # 데이터 준비
    class_stats = []
    for class_idx in range(num_classes):
        class_key = str(class_idx)
        ious = [data["class_ious"].get(class_key, 0) for data in class_ious_list 
                if class_key in data["class_ious"]]
        avg_iou = np.nanmean(ious) if ious else 0
        
        total_gt = sum(data["gt_pixel_counts"].get(class_key, 0) for data in class_ious_list)
        total_pred = sum(data["pred_pixel_counts"].get(class_key, 0) for data in class_ious_list)
        
        class_stats.append({
            'class_idx': class_idx,
            'avg_iou': avg_iou,
            'total_gt_pixels': total_gt,
            'total_pred_pixels': total_pred
        })
    
    df_stats = pd.DataFrame(class_stats)
    
    # 이미지별 평균 IoU
    image_ious = [data["mean_iou"] for data in class_ious_list]
    
    # 1. 클래스별 평균 IoU 바 차트
    fig1 = px.bar(df_stats, x='class_idx', y='avg_iou', 
                 title='클래스별 평균 IoU',
                 labels={'class_idx': '클래스 인덱스', 'avg_iou': '평균 IoU'})
    
    # 2. GT 픽셀 수 vs 예측 픽셀 수 (로그 스케일)
    fig2 = px.scatter(df_stats, x='total_gt_pixels', y='total_pred_pixels', 
                      color='avg_iou', size='avg_iou', size_max=20,
                      hover_name='class_idx', log_x=True, log_y=True,
                      title='클래스별 GT vs 예측 픽셀 수 (로그 스케일)',
                      labels={'total_gt_pixels': 'GT 픽셀 수 (로그 스케일)', 
                              'total_pred_pixels': '예측 픽셀 수 (로그 스케일)',
                              'avg_iou': '평균 IoU'})
    
    # 3. 이미지별 mIoU 히스토그램
    fig3 = px.histogram(image_ious, title='이미지별 mIoU 분포',
                       labels={'value': 'mIoU', 'count': '이미지 수'}, 
                       nbins=30)
    
    # 4. 클래스별 픽셀 분포 원 그래프
    fig4 = px.pie(df_stats, values='total_gt_pixels', names='class_idx',
                  title='클래스별 GT 픽셀 분포')
    
    # 대시보드 레이아웃 구성 (2x2 그리드)
    dashboard = make_subplots(rows=2, cols=2, 
                              subplot_titles=['클래스별 평균 IoU', 
                                             'GT vs 예측 픽셀 수',
                                             'mIoU 분포',
                                             '클래스별 GT 픽셀 분포'],
                              specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                                     [{'type': 'histogram'}, {'type': 'pie'}]])
    
    # 첫 번째 그래프 추가
    for trace in fig1.data:
        dashboard.add_trace(trace, row=1, col=1)
    
    # 두 번째 그래프 추가
    for trace in fig2.data:
        dashboard.add_trace(trace, row=1, col=2)
    
    # 세 번째 그래프 추가
    for trace in fig3.data:
        dashboard.add_trace(trace, row=2, col=1)
    
    # 네 번째 그래프 추가
    for trace in fig4.data:
        dashboard.add_trace(trace, row=2, col=2)
    
    # 대시보드 제목 및 레이아웃 설정
    dashboard.update_layout(
        title_text='세그멘테이션 결과 대시보드',
        height=800,
        showlegend=False
    )
    
    # 대시보드를 HTML 파일로 저장
    dashboard.write_html(filename)
    
    print(f"인터랙티브 대시보드가 '{filename}' 파일로 저장되었습니다. 웹 브라우저로 열어 확인하세요.")

# 함수 호출
create_dashboard(class_ious_list, num_classes, avg_class_ious)