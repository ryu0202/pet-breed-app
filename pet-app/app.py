import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError


st.set_page_config(page_title="반려동물 품종 분류", page_icon="🐾", layout="wide")


@st.cache_resource
def get_predictor(path, modified_time):
    from predict import load_model
    return load_model(path)

with st.sidebar:
    st.header("반려동물 품종 분류하기")
    st.divider()
    st.caption("동물 한 마리가 선명하게 보이는 사진을 사용해 주세요.")
    st.subheader("분류 가능한 품종")
    st.markdown("**개:** 비글, 치와와, 퍼그, 시바견, 사모예드")
    st.markdown("**고양이:** 벵갈, 페르시안, 랙돌, 샴, 스핑크스")

st.title("반려동물 품종 분류하기")
st.write("사진 속 반려동물의 품종을 확인해 보세요.")
st.divider()

image_column, result_column = st.columns([1.2, 1], gap="large")
photo = None
with image_column:
    st.subheader("사진")
    uploaded = st.file_uploader("사진 선택", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        try:
            photo = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            st.image(photo, use_container_width=True)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            photo = None
            st.error("사진을 읽을 수 없습니다. 다른 JPG 또는 PNG 파일을 선택해 주세요.")

with result_column:
    st.subheader("분류 결과")
    if photo is not None:
        checkpoint = Path(__file__).parent / "models" / "best.pt"
        try:
            from predict import predict_image
            with st.spinner("사진을 분석하고 있습니다."):
                model, classes, transform = get_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns)
                results = predict_image(photo, model, classes, transform)
            for rank, result in enumerate(results, start=1):
                with st.container(border=True):
                    st.markdown(f"**{rank}순위 · {result['korean']}**")
                    st.write(f"모델 예측값 {result['score']:.1%}")
                    st.progress(result["score"])
            st.caption("예측값은 모델의 점수이며 실제 정답 확률을 보장하지 않습니다. 대상 10개 품종에 대한 후보를 제공합니다.")
        except FileNotFoundError:
            st.error("모델 파일을 찾을 수 없습니다. models/best.pt를 확인해 주세요.")
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            st.error("사진을 분석하지 못했습니다. 실행 환경과 모델 파일을 확인해 주세요.")
            with st.expander("오류 내용"):
                st.code(str(exc))
