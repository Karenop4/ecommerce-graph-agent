# frontend/Dockerfile
FROM node:22-alpine

WORKDIR /app

# Copiar archivos de dependencias
COPY package.json package-lock.json ./

# Instalar dependencias
RUN npm install

# Copiar el resto del código
COPY . .

# Exponer el puerto de Vite (por defecto 5173)
EXPOSE 5173

# IMPORTANTE: --host permite acceder desde fuera del contenedor
CMD ["npm", "run", "dev", "--", "--host"]