# Stage 1: Build the Vite app
FROM node:20-alpine AS build
WORKDIR /app

# Build arg: API base URL baked into JS at compile time.
# Default "" means frontend uses "/api" (relative path) — works with a reverse proxy.
# Override with Cloud Run API URL for GCP deploy.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE

COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve static files with nginx
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
