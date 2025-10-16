<template>
  <div class="participant-chip" :title="title || label">
    <div class="chip-avatar" :style="{ backgroundColor: color }">
      {{ initials }}
    </div>
    <span class="chip-name">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
  import { computed } from "vue";
  interface Props {
    label: string;
    title?: string;
  }
  const props = defineProps<Props>();

  const getSpeakerInitials = (name: string): string => {
    if (!name) return "";
    const words = name.trim().split(/\s+/);
    if (words.length === 1) return words[0].substring(0, 2).toUpperCase();
    return words
      .slice(0, 2)
      .map((w) => w[0])
      .join("")
      .toUpperCase();
  };

  const getSpeakerColor = (name: string): string => {
    if (!name) return "#646cff";
    let hash = 0;
    for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
    const hue = Math.abs(hash % 360);
    return `hsl(${hue}, 65%, 55%)`;
  };

  const initials = computed(() => getSpeakerInitials(props.label));
  const color = computed(() => getSpeakerColor(props.label));
</script>

<style scoped>
.participant-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.chip-avatar {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  font-weight: 800;
  color: #fff;
}

.chip-name {
  font-size: 0.8rem;
  font-weight: 600;
  color: #ddd;
}
</style>
