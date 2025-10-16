<template>
  <div
    class="participant-chip"
    :class="{ interactive, active: isActive }"
    :title="title || label"
    @click="onClick"
    @keydown.enter.prevent="onClick"
    @keydown.space.prevent="onClick"
    :role="interactive ? 'button' : undefined"
    :tabindex="interactive ? 0 : undefined"
  >
    <span v-if="icon" class="chip-icon" aria-hidden="true">{{ icon }}</span>
    <div
      v-else-if="showAvatar"
      class="chip-avatar"
      :style="{ backgroundColor: color }"
    >
      {{ initials }}
    </div>
    <span class="chip-name">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
  import { computed } from "vue";
  const emit = defineEmits<{ (e: "click"): void }>();

  interface Props {
    label: string;
    title?: string;
    icon?: string;
    isActive?: boolean;
    interactive?: boolean;
    showAvatar?: boolean;
  }
  const props = withDefaults(defineProps<Props>(), {
    isActive: false,
    interactive: false,
    showAvatar: true,
  });

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
    for (let i = 0; i < name.length; i++)
      hash = name.charCodeAt(i) + ((hash << 5) - hash);
    const hue = Math.abs(hash % 360);
    return `hsl(${hue}, 65%, 55%)`;
  };

  const initials = computed(() => getSpeakerInitials(props.label));
  const color = computed(() => getSpeakerColor(props.label));

  const onClick = () => {
    if (props.interactive) emit("click");
  };
</script>

<style scoped>
  .participant-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.6rem 0.25rem 0.25rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .participant-chip.interactive {
    cursor: pointer;
    user-select: none;
    transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
  }
  .participant-chip.interactive:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.18);
  }
  .participant-chip.active {
    border-color: rgba(29, 185, 84, 0.45);
    box-shadow: 0 0 0 2px rgba(29, 185, 84, 0.15) inset;
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

  .chip-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 6px;
    background: #1a73e8;
    color: #fff;
    font-size: 0.65rem;
    line-height: 1;
    transform: translateY(-0.5px);
  }

  .chip-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #ddd;
  }
</style>
