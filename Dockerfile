# build stage
FROM node:lts-alpine as build-stage
WORKDIR /app

# 安装 pnpm
RUN npm install -g pnpm

# 复制 package.json 和 pnpm-lock.yaml（如果有的话）
COPY package*.json pnpm-lock.yaml* ./

# 使用 pnpm 安装依赖
RUN pnpm install

COPY . .
RUN pnpm run docs:build

# production stage
FROM nginx:stable-alpine as production-stage
COPY --from=build-stage /app/docs/.vuepress/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]